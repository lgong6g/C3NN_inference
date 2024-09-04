import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import equivariant steerable cnn
import escnn.gspaces
import escnn.nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy

from tqdm import tqdm
# Checking if you have a GPU and CUDA set up -- if not, training will take longer by running on the CPU
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
# device = 'cpu'
# Set the plotting style
#plt.style.use('seaborn-poster')

from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn

from os import listdir
from os.path import isfile, join

from copy import deepcopy

from getdist import MCSamples, plots

import pickle as pk

from hyperparameters import hparams

ens = 5#20 # Turn this up # right now 5*5000 blocks 30GB, so with 80GB GPU should be able to train 5*all
path = '/project/ag-weller/Kai.Lehman/gaussian_fields/'
onlyfiles = [f[:-4] for f in listdir(path) if f[-3:] == 'npy']
onlyfiles_250_by_250 = np.array([file for file in onlyfiles if '250_by_250' in file])[:5000]
#onlyfiles_split = np.array([s.split('_') for s in onlyfiles])
onlyfiles_250_by_250_split = np.array([s.split('_') for s in onlyfiles])[:5000]
stds = onlyfiles_250_by_250_split[:,8].astype(np.float32)
corrlens = onlyfiles_250_by_250_split[:,10].astype(np.float32)

#imgs = np.zeros((ens*len(onlyfiles),100,100))
imgs = np.zeros((ens*len(onlyfiles_250_by_250),250,250))
print('Reading files\n')
for i, name in enumerate(onlyfiles_250_by_250):
    img_stack = np.load(path+name+'.npy')
    imgs[i*ens:(i+1)*ens] = img_stack[:ens]
    if i%500 ==0: print(i, 'read')


print('Moving to device')
#imgs = from_numpy(np.reshape(imgs.astype(np.float32),((ens*len(onlyfiles),1,100,100)))).to('cuda')
imgs = from_numpy(np.reshape(imgs.astype(np.float32),((ens*len(onlyfiles_250_by_250),1,250,250)))).to('cuda')

stds_tiled = (np.ones((ens,len(stds)),dtype=np.float32)*stds).T.flatten()
corrlens_tiled = (np.ones((ens,len(corrlens)),dtype=np.float32)*corrlens).T.flatten()

print('Building Data Vector')
# building one theta vector
theta = from_numpy(np.vstack((stds_tiled,corrlens_tiled)).T).to('cuda')
print(theta.shape)

# set up C3NN
bandlimit = 6
gspace = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=bandlimit)


""" Implements the non-linear convolution operation using the recursive formula."""
class NonlinearConv(nn.Module):
    def __init__(self, in_channels, num_filts, filter_size, order):
        super().__init__()
        self.order = order
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size

        """
        # Initialize filter similar as to how Pytorch initializes nn.Conv2d, but all positive
        # This is the old way of initializing C3NN filters without rotational equivariance/invariance
        weight_sample_k = 1.0 / np.sqrt(filter_size**2 * in_channels)
        init_filt = (2 * weight_sample_k * torch.rand(num_filts, in_channels, filter_size, filter_size, dtype=torch.float32))
        self.conv_filt = torch.nn.Parameter(init_filt)
        # Initialize some slices of the filter to be 0
        with torch.no_grad():
            self.conv_filt[:, :, int(self.filter_size/2)-2:int(self.filter_size/2)+3, int(self.filter_size/2)-2:int(self.filter_size/2)+3]=0.0
        """

        # Initialize the escnn filter
        feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.trivial_repr])
        feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.trivial_repr])
        # THIS LAYER KILLS DEEPCOPY
        self.conv = escnn.nn.R2Conv(in_type=feat_type_in, out_type=feat_type_out, bias=False, kernel_size=self.filter_size)

        # to enable deepcopy
        self.detached = False

    def forward(self, x):
        # We will work with strictly positive filters for easier interpretation!
        # foll
        if self.detached == False:
            self.abs_filt = self.conv.expand_parameters()[0].abs()
        elif self.detached == True:
            pass

        # Convolutions of powers of the input with powers of the absolute valued filter
        conv_powers = [F.conv2d(x ** k, self.abs_filt ** k, stride=1, padding=int((self.filter_size-1)/2.0)) for k in range(1, self.order + 1)]

        #troubleshooting:
        #conv_powers = [self.convnormal(x ** k) for k in range(1, self.order + 1)]

        # C_0 is defined to be one everywhere
        corrs = [torch.ones_like(conv_powers[0])]

        # Compute each order of C_n recursively
        for n in range(1, self.order + 1):
            Cn = torch.zeros_like(conv_powers[0])
            for k in range(n):
                Cn += (-1) ** k * conv_powers[k] * corrs[n - k - 1]
            Cn /= n
            corrs.append(Cn)

        # Stack correlator maps along the channel dimension, dropping the zeroth order (C_0)
        # Here the channel dimension refers to the number of channels
        #return torch.cat(corrs[1:], dim=-3)
        return torch.cat(corrs[1:], dim=-3)

    def _detach(self):
        del self.conv
        self.abs_filt = self.abs_filt.detach()
        self.detached = True


""" Does a simple average over the last two dimensions of the input."""
class Mean2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean((-2, -1))


""" The full CCNN is Input -> NonlinearConv -> Spatial Avg -> BatchNorm -> Logistic. The way it is constructed
    here assumes a binary classification task, but this can be extended to the multiclass case."""
class CCNN(nn.Module):
    def __init__(self, in_channels, num_filts, filter_size, order, absbeta=False, slice=False, pool=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order
        self.absbeta = absbeta

        # the model architecture with escnn
        self.corr_conv = NonlinearConv(self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d()
        # I preserve the batch normalization layer here so that we can scale the compressed summary statistics to the same order of magnitude
        # This may help the numerical stability of inference in the 2nd part
        self.bn = nn.BatchNorm1d(num_filts * order, affine=True, eps=3.185289e-35, dtype=torch.float32)


        #troubleshooting
        #self.conv = Conv2d(1,1,11)

    def forward(self, x, chop=True):

        x = self.corr_conv(x)
        #x = self.conv(x)
        x = self.spatial_mean(x)
        if chop == False:
            x = self.bn(x)


        return x

    def detach(self):
        self.corr_conv._detach()


#=== 2) Set up some things for training ===#
# Number of times the model will see the full dataset
#epochs = 5
# The initial learning rate -- an important hyperparameter
#lr = 0.505631
#lr = 0.5
# The strength of the regularization loss -- we can make this very strong in these easy toy examples
#gamma = 0.030383
# The number of input channels
in_channels = 1
# The number of filters to learn in our model
num_filts = 1
# The spatial length of the filters
filter_size = 11
# The highest order of correlation measured
order = 2

ccnn = CCNN(in_channels=in_channels, num_filts=num_filts, filter_size=filter_size, order=order, absbeta=False).to(device='cuda')

# Set up prior for sbi
lower = np.array([1.,5.])
upper = np.array([51.,23.])
#lower = np.array([.8,3.])
#upper = np.array([4.,30.])
prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device='cuda')


#set up sbi architecture
density_estimator_custom = posterior_nn(model="maf", hidden_features=hparams.flow_hidden_features, num_transforms=hparams.flow_num_transforms, device='cuda', embedding_net=ccnn)
inference = SNPE_C(prior, device='cuda', density_estimator=density_estimator_custom)

# joint training
density_estimator = inference.append_simulations(theta, imgs).train(learning_rate=hparams.learning_rate,training_batch_size=hparams.flow_training_batch_size)
# DANGER DANGER
ccnn.detach()
# now deepcopy works:
test = deepcopy(density_estimator._embedding_net)

if hparams.save:
    with open(f"{hparams.name_model()}_inference.pkl", "wb") as handle:
                pk.dump(inference, handle)
    with open(f"{hparams.name_model()}_density_estimator.pkl", "wb") as handle:
                pk.dump(deepcopy(density_estimator), handle)
    print('Model saved')

train_loader, val_loader = inference.get_dataloaders(
            #start_idx,
            training_batch_size=1,
            #validation_fraction,
            resume_training=False,
            #dataloader_kwargs=dataloader_kwargs,
            dataloader_kwargs={"drop_last": False},
)

theta_test, img_test, _ = next(iter(val_loader))

# build posterior and sample
index = 2000
posterior = inference.build_posterior(deepcopy(density_estimator))
posterior_samples = posterior.sample((100000,), x=img_test).cpu()

names = np.array(['\sigma','corrlen'])
params = theta_test[0].cpu()


samples = MCSamples(samples=posterior_samples.numpy(), names=names, labels=names, settings={'ignore_rows':.2},label='sbi, Likelihood-Free')
g = plots.get_subplot_plotter()
#g.settings.lab_fontsize = 17
#g.settings.axes_fontsize = 14.3
#g.settings.legend_fontsize = 14.5
g.triangle_plot([samples], names, title_limit=1, filled=True,
                     markers={'\sigma':params[0],'corrlen':params[1]},
                     marker_args={'c':'black'})

plt.show()
plt.savefig('test_250_by_250.png')

# let's do a more statistical test over the entire validation set

no_val_sims = len(val_loader)
posterior_samples_all = np.zeros((no_val_sims,100000,2))
params = np.zeros((no_val_sims,2))
for i, (param, observation, _) in enumerate(val_loader):
    params[i] = param.cpu()
    observation.to('cuda')
    posterior_samples_all[i] = posterior.sample((100000,), x=observation).cpu()
    if i%10 == 0:
        print(i)

no_val_sims = 100
posterior_samples_all = np.zeros((no_val_sims,100000,2))
params = np.zeros((no_val_sims,2))
for i, (param, observation, _) in enumerate(val_loader):
    params[i] = param.cpu()
    observation.to('cuda')
    posterior_samples_all[i] = posterior.sample((100000,), x=observation).cpu()
    if i%10 == 0:
        print(i)
    if i == no_val_sims-1:
        break

fig, ax = plt.subplots(2)
parts0 = ax[0].violinplot(posterior_samples_all[:no_val_sims,:,0].T-params[:no_val_sims,0],params[:no_val_sims,0],showextrema=False,quantiles=np.tile([.16,.84],(no_val_sims,1)).T)
parts1 = ax[1].violinplot(posterior_samples_all[:no_val_sims,:,1].T-params[:no_val_sims,1],params[:no_val_sims,1],showextrema=False,quantiles=np.tile([.16,.84],(no_val_sims,1)).T)

for i, axis in enumerate(ax):
    axis.axhline(0,color='black')
    axis.set_xlabel(f'True {names[i]}')
    axis.set_ylabel(f'Predicted - True {names[i]}')
plt.savefig('violin_250_by_250.png')
