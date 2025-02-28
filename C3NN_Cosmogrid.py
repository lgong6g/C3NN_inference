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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# Checking if you have a GPU and CUDA set up -- if not, training will take longer by running on the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn
from os import listdir
from os.path import isfile, join
from copy import deepcopy
from hyperparameters import hparams
import cloudpickle as cldpk


training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
training_data = torch.tensor(training_data, dtype=torch.float32)

param_names = ['Om', 'sigma8']
training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
training_parameters = torch.tensor(training_parameters, dtype=torch.float32)

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
        
        # Initialize the escnn filter
        feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.trivial_repr])
        feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.trivial_repr])
        self.conv = escnn.nn.R2Conv(in_type=feat_type_in, out_type=feat_type_out, bias=False, kernel_size=self.filter_size) 
        
        
    def forward(self, x):
        # We will work with strictly positive filters for easier interpretation!
        self.abs_filt = self.conv.expand_parameters()[0].abs()
        
        # Convolutions of powers of the input with powers of the absolute valued filter
        conv_powers = [F.conv2d(x ** k, self.abs_filt ** k, stride=1, padding=int((self.filter_size-1)/2.0)) for k in range(1, self.order + 1)]
        
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


""" Does a simple average over the last two dimensions of the input."""
class Mean2d(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.mean((-2, -1))


""" The full CCNN is Input -> NonlinearConv -> Spatial Avg -> BatchNorm -> Logistic. The way it is constructed
    here assumes a binary classification task, but this can be extended to the multiclass case."""
class C3NN(nn.Module):
    def __init__(self, in_channels, num_filts, filter_size, order, absbeta=False):
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
        self.parallel = torch.vmap(self.forward_single, in_dims=0, out_dims=0)
        #troubleshooting
        #self.conv = Conv2d(1,1,11)
    
    def forward_single(self, x, chop=True):    
        
        x = self.corr_conv(x)
        #x = self.conv(x)
        x = self.spatial_mean(x)
        if chop == False:
            x = self.bn(x)
        return x
    
    def forward(self, x):
        return torch.mean(self.parallel(x), 1)


#=== 2) Set up some things for training ===#
# The number of input channels 
in_channels = 4
# The number of filters to learn in our model
num_filts = 2
# The spatial length of the filters
filter_size = 37
# The highest order of correlation measured
order = 3

c3nn = C3NN(in_channels=in_channels, num_filts=num_filts, filter_size=filter_size, order=order, absbeta=False).to(device='cuda')

# Set up prior for sbi
lower = np.array([0.1, 0.4])
upper = np.array([0.5, 1.4])
prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device='cuda')


#set up sbi architecture
density_estimator_custom = posterior_nn(model="maf", hidden_features=hparams.flow_hidden_features, num_transforms=hparams.flow_num_transforms, device='cuda', embedding_net=c3nn)
inference = SNPE_C(prior, device='cuda', density_estimator=density_estimator_custom)

# joint training
density_estimator = inference.append_simulations(training_parameters, training_data, data_device='cuda').train(learning_rate=hparams.learning_rate,training_batch_size=hparams.flow_training_batch_size,max_num_epochs=2e4)

with open('density_estimator_cosmogrid_1250_nodes_Om_sigma8_order3_filter2_size37.pickle', 'wb') as f:
    cldpk.dump(density_estimator, f)

with open('inference_cosmogrid_1250_nodes_Om_sigma8_order3_filter2_size37.pickle', 'wb') as f:
    cldpk.dump(inference, f)

