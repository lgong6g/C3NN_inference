import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model

# import equivariant steerable cnn
import escnn.gspaces
import escnn.nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
#from data_util import _vflip, _hflip, _rot
# Checking if you have a GPU and CUDA set up -- if not, training will take longer by running on the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read in FLASK Gaussian and Lognormal random field maps #
#log_example = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z16_only/lognormal_data_17760_maps_120_by_120pixels_4_by_4deg_zs16_gnom_projection.npy')[:2000]
#grf_example = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z16_only/gaussian_data_17760_maps_120_by_120pixels_4_by_4deg_zs16_gnom_projection.npy')[:2000]

#log_example_lambda1 = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/lognormal_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy')[:,0,:,:]
#log_example_lambda2 = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/lognormal_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_new_lambda_gnom_projection.npy')[:,0,:,:]

log_example = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/lognormal_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy')[:5000,1,:,:]
grf_example = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/gaussian_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy')[:5000,1,:,:]

###### Load the GSTools maps with Gaussian (as well as others) covariance model #######

#grf_example_s = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/grf_data_10000_maps_120_by_120_pixels_var_1.0_corrlen_5.0.npy')[:5000]
#erf_example_s = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/erf_data_10000_maps_120_by_120_pixels_var_1.0_corrlen_5.0_5.0.npy')
#grf_example_l = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/grf_data_10000_maps_120_by_120_pixels_var_1.0_corrlen_20.0.npy')[:5000]

#### Load the Cosmogrid simulation maps with intrinsic alignment signal ####

#kappa_no_IA = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_plus_IA_NLA_A0.0_alpha_1.66_data_11600_maps_200_by_200pixels_20_by_20deg_4_des_zsbin_gnom_projection.npy')
#kappa_IA_Aamp = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_plus_IA_NLA_A0.36_alpha_1.66_data_11600_maps_200_by_200pixels_20_by_20deg_4_des_zsbin_gnom_projection.npy')
# kappa_IA_A8 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_plus_IA_NLA_A8.0_data_11600_maps_200_by_200pixels_20_by_20deg_des_zbin2_4_gnom_projection.npy')[:,1,:,:]
#kappa_IA_A1e4 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_plus_IA_NLA_A1e4_data_11600_maps_200_by_200pixels_20_by_20deg_des_zbin2_4_gnom_projection.npy')[:,1,:,:]

#### Load the Takahashi N-body maps and the flask log-normal maps with the same cosmology ####
#t17 = np.load('/e/ocean1/users/lgong/CCNN/takahashi/kappa_data_6264_maps_200_by_200pixels_20_by_20deg_zs16_gnom_projection.npy')[:,0,:,:]
#flask = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/lognormal_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy')[:6264,1,:,:]

"""Load snapshots following a spatial random distribution"""
def load_srf(num_snaps, size, srf_data):
    snaps = torch.zeros(num_snaps, 1, size, size, dtype=torch.float32)
    data = torch.tensor(srf_data, dtype=torch.float32)
    # Set the channel to random gaussian distribution
    snaps[:, 0, :, :] = data
    # Split into a train and val set
    num_train = int(0.9 * num_snaps)
    return snaps[:num_train], snaps[num_train:]

def load_multi_channel_data(num_snaps, srf_data):
    snaps = torch.tensor(srf_data, dtype=torch.float32)
    # Split into a train and val set
    num_train = int(0.9 * num_snaps)
    return snaps[:num_train], snaps[num_train:]

#train_gaussian, val_gaussian = load_srf(1523, 200, grf_example)
#train_log, val_log = load_srf(1523, 200, log_example)
train_grf, val_grf = load_srf(5000, 200, grf_example)
train_log, val_log = load_srf(5000, 200, log_example)

"""Assembles tensors of snapshots into torch.DataLoaders to use during training"""
def make_datasets(train_A, train_B, val_A, val_B, batch_size=256):
    train_tensors = torch.cat([train_A, train_B])
    val_tensors = torch.cat([val_A, val_B])
    # Class A is labeled with a 0, class B is labeled with a 1
    train_labels = torch.cat([
        torch.zeros(len(train_A), dtype=torch.int64),
        torch.ones(len(train_B), dtype=torch.int64)
    ])
    val_labels = torch.cat([
        torch.zeros(len(val_A), dtype=torch.int64),
        torch.ones(len(val_B), dtype=torch.int64)
    ])
    
    train_dataset = torch.utils.data.TensorDataset(train_tensors, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_tensors, val_labels)
    
    pin_memory = True if torch.cuda.is_available() else False
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )
    return train_loader, val_loader

# Set up the continuous planar rotation group SO(2)
bandlimit = 6
gspace = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=bandlimit)
#feat_type_in = escnn.nn.FieldType(gspace, [gspace.irrep(0)])
#feat_type_out = escnn.nn.FieldType(gspace, [gspace.irrep(0)])

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
        weight_sample_k = 1.0 / np.sqrt(filter_size**2 * in_channels)
        init_filt = (2 * weight_sample_k * torch.rand(num_filts, in_channels, filter_size, filter_size, dtype=torch.float32))
        self.conv_filt = torch.nn.Parameter(init_filt)
        """
        # Initialize the escnn filter
        #feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.irrep(0)])
        #feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.irrep(0)])
        #self.conv = escnn.nn.R2Conv(in_type=feat_type_in, out_type=feat_type_out, bias=False, kernel_size=self.filter_size, padding=int((self.filter_size-1)/2.0))

        feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.trivial_repr])
        feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.trivial_repr])
        self.conv = escnn.nn.R2Conv(in_type=feat_type_in, out_type=feat_type_out, bias=False, kernel_size=self.filter_size) 
        #with torch.no_grad():
        #    self.conv.weights[:3] = 0.0
    
    def forward(self, x):
        # inputs to transform according to its input type and is guaranteed to return feature fields associated with its output type.
        #x = feat_type_in(x)

        #assert isinstance(x.tensor, torch.Tensor)
        
        # We will work with strictly positive filters for easier interpretation!
        self.abs_filt = self.conv.expand_parameters()[0].abs()
        #abs_filt = self.conv.expand_parameters()[0].abs()
        #abs_filt = self.conv_filt.abs()

        # Convolutions of powers of the input with powers of the absolute valued filter
        conv_powers = [F.conv2d(x ** k, self.abs_filt ** k, stride=1, padding=int((self.filter_size-1)/2.0)) for k in range(1, self.order + 1)]
        #conv_powers = [F.conv2d(x ** k, abs_filt ** k, stride=1, padding=int((self.filter_size-1)/2.0)) for k in range(1, self.order + 1)]

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
        return torch.cat(corrs[1:], dim=-3)

""" Return the input """
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

""" Does a simple average over the last two dimensions of the input."""
class Mean2d(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.mean((-2, -1))

""" Concatenate the one-column output from the logistic regression layer with a column of 0 to enable cross entropy loss calculation"""
class probability_concatenation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat((x, torch.zeros_like(x)), dim=-1)

""" 
The full CCNN is Input -> NonlinearConv -> Spatial Avg -> BatchNorm -> Logistic. The way it is constructed here assumes a binary classification task, but this can be extended to the multiclass case.

Here the slice and pool variable indicate whether to carry out the symmetry slicing and pooling operations or not
"""
class CCNN(nn.Module):
    def __init__(self, in_channels, num_filts, filter_size, order, absbeta=False, slice=False, pool=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order
        self.absbeta = absbeta
        """
        self.slice = slice
        self.pool = pool

        if self.slice:
            self.symslice = SymSlice()
        else:
            self.symslice = Identity()
        self.corr_conv = NonlinearConv(in_channels, num_filts, filter_size, order)
        if self.slice:
            self.sympool = SymPool()
        else:
            self.sympool = Identity()
        self.spatial_mean = Mean2d()
        self.bn = nn.BatchNorm1d(num_filts * order, affine=True, eps=3.185289e-35, dtype=torch.float32)
        self.logistic_fc = nn.Linear(num_filts * order, 1, dtype=torch.float32)
        # If absbeta is set, force the beta_alpha^(n) parameters to be strictly positive
        # The effect of this is that the model can only pick up features which "give evidence for" class A
        if self.absbeta:
            self.logistic_fc.weight.data.abs_()
        self.concat = probability_concatenation()
        """

        # the model architecture with escnn
        self.corr_conv = NonlinearConv(self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d()
        self.bn = nn.BatchNorm1d(self.num_filts * self.order, affine=True, eps=3.185289e-35, dtype=torch.float32)
        self.logistic_fc = nn.Linear(self.num_filts * self.order, 1, dtype=torch.float32)
        # If absbeta is set, force the beta_alpha^(n) parameters to be strictly positive
        # The effect of this is that the model can only pick up features which "give evidence for" class A
        self.concat = probability_concatenation()
        
    def forward(self, x, chop=False):
        """
        x = self.symslice(x)
        x = self.corr_conv(x)
        x = self.sympool(x)
        x = self.spatial_mean(x)
        x = self.bn(x)
        x = self.logistic_fc(x)
        # Treat the output as if logit 0 has the value predicted by model, logit 1 has logit 0
        pred = self.concat(x)
        """
        x = self.corr_conv(x)
        x = self.spatial_mean(x)
        if chop == False:
            x = self.bn(x)
            x = self.logistic_fc(x)
            if self.absbeta:
                self.logistic_fc.weight.data.abs_()
            # Treat the output as if logit 0 has the value predicted by model, logit 1 has logit 0
            x = self.concat(x)

        return x

#=== 1) Create SRF DataLoaders and load the optimized optuna object===#

train_loader, val_loader = make_datasets(train_grf, train_log, val_grf, val_log)
#train_loader, val_loader = make_datasets(train_grf_s, train_grf_l, val_grf_s, val_grf_l)
with open('../../study_saves/toy_models/ccnn_study_flask_1_channel_1_filter_31pix_t17_5e3maps_gaussian_vs_lognormal_zs16_200_pixels_order4_escnn.pickle', 'rb') as f:
    study = pickle.load(f)

#=== 2) Set up some things for training ===#
# Number of times the model will see the full dataset
epochs = 100
# The initial learning rate -- an important hyperparameter
lr = study.best_trial.params['lr']
# The strength of the regularization loss -- we can make this very strong in these easy toy examples
gamma = study.best_trial.params['regularization_gamma']
# The number of input channels 
in_channels = 1
# The number of filters to learn in our model
num_filts = 1
# The spatial length of the filters
#filter_size = study.best_trial.params['filter_size']
filter_size = 31
# The highest order of correlation measured
#order = study.best_trial.params['corr_order']
order = 4
# Make our model -- for simplicity, let's train a two-filter model
ccnn = CCNN(in_channels=in_channels, num_filts=num_filts, filter_size=filter_size, order=order).to(device=device)

# The optimizer
if study.best_trial.params['optimizer'] == "Adam":
    optimizer = torch.optim.Adam(ccnn.parameters(), lr=lr)
elif study.best_trial.params['optimizer'] == "RMSprop":
    optimizer = torch.optim.RMSprop(ccnn.parameters(), lr=lr)
else:
    optimizer = torch.optim.SGD(ccnn.parameters(), lr=lr)

# A learning schedule which will ramp down the learning rate as we train
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, study.best_trial.params['lr_scheduler_ratio'])
# Our loss function
criterion = nn.CrossEntropyLoss()

"""Performs one training epoch, feeding an entire DataLoader through a model,
   performing gradient descent and keeping track of the average loss and accuracy."""
def train(model, optimizer, criterion, loader, gamma, device='cuda'):
    model.train()
    loss_total = 0.0
    accuracy_total = 0.0
    for inputs, labels in loader:
        # Zero out gradients from the previous iteration
        optimizer.zero_grad()
        
        # Send the inputs and outputs to the GPU (if using one)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Feed the input through the network, measure the loss
        pred = model(inputs)
        loss = criterion(pred, labels)
        
        # Add our regularization loss on the filter weights
        #l1_loss = model.corr_conv.conv_filt.norm(1)
        l1_loss = model.corr_conv.abs_filt.norm(1)
        loss += gamma * l1_loss

        # Backprop gradients
        loss.backward()

        """
        # fix the gradient of the pixels within the scale cut
        for i, param in enumerate(model.parameters()):
            if i == 0:
                param.grad[:3] = torch.zeros_like(param.grad[:3])
                #param.grad[:,:,int(model.filter_size/2)-2:int(model.filter_size/2)+3, int(model.filter_size/2)-2:int(model.filter_size/2)+3] = torch.zeros_like(param.grad[:,:,int(model.filter_size/2)-2:int(model.filter_size/2)+3, int(model.filter_size/2)-2:int(model.filter_size/2)+3])
        """
        
        # Update model parameters by one gradient descent step
        optimizer.step()

        # Do some bookkeeping of the current loss and accuracy
        loss_total += loss.item()
        pred_classes = pred.argmax(dim=1).cpu()
        labels = labels.cpu()
        accuracy_total += torch.count_nonzero(pred_classes == labels).item() / len(pred_classes)

    avg_loss = loss_total / len(loader)
    avg_acc = accuracy_total / len(loader)
    return avg_loss, avg_acc

"""Performs one validation epoch, feeding an entire DataLoader through a model, and
   calculating the average loss and accuracy."""
def val(model, criterion, loader, gamma, device='cuda'):
    model.eval()
    loss_total = 0.0
    accuracy_total = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            pred = model(inputs)
            loss = criterion(pred, labels)

            #l1_loss = model.corr_conv.conv_filt.norm(1)
            l1_loss = model.corr_conv.abs_filt.norm(1)
            loss += gamma * l1_loss

            loss_total += loss.item()
            pred_classes = pred.argmax(dim=1).cpu()
            labels = labels.cpu()
            accuracy_total += torch.count_nonzero(pred_classes == labels).item() / len(pred_classes)
            
    avg_loss = loss_total / len(loader)
    avg_acc = accuracy_total / len(loader)
    return avg_loss, avg_acc

#=== 3) The main training loop! ===#
# Some lists to keep track of losses and accuracies
train_losses, val_losses = [], []
train_accs, val_accs = [], []
for epoch in tqdm(range(epochs), desc="Epoch"):
    # Train
    train_loss, train_acc = train(
        ccnn, optimizer, criterion, train_loader,
        gamma=gamma, device=device
    )
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Val
    val_loss, val_acc = val(
        ccnn, criterion, val_loader,
        gamma=gamma, device=device
    )
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Step the learning rate schedule
    lr_scheduler.step()

torch.save(ccnn.state_dict(), '../../../trained_model/toy_models/ccnn_flask_t17_gaussian_vs_lognormal_zs16_5e3maps_1_channel_1_filter_31pix_classification_200pixels_order4_hypertuned_escnn.pt')

### How to load the model saved in this way
# model = MyModelDefinition(args)
# model.load_state_dict(torch.load('load/from/path/model.pth'))

