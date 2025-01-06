import os
import time
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
import torch.optim as optim
import torch.nn.functional as F

import optuna
from optuna.trial import TrialState

from tqdm import tqdm
#plt.style.use('seaborn-poster')

###### Load the N-body simulation maps ######

#mv00_example = np.load('/e/ocean1/users/lgong/CCNN/massivenus_data/mv0.0_omega0.3/kappa_maps_zs10_512_by_512pixels_3.5_by_3.5deg.npy')
#mv01_example = np.load('/e/ocean1/users/lgong/CCNN/massivenus_data/mv0.1_omega0.3/kappa_maps_zs10_512_by_512pixels_3.5_by_3.5deg.npy')

#############################################

###### Load the FLASK Gaussian and Log-normal random fields with the same angular power spectrum #######

grf_example = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/gaussian_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy')[:5000,1,:,:]
log_example = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/lognormal_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy')[:5000,1,:,:]

#log_example_lambda1 = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/lognormal_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy')[:,0,:,:]
#log_example_lambda2 = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/lognormal_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_new_lambda_gnom_projection.npy')[:,0,:,:]

#grf_example = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z16_only/gaussian_data_17760_maps_120_by_120pixels_4_by_4deg_zs16_gnom_projection.npy')[:2000]
#log_example = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z16_only/lognormal_data_17760_maps_120_by_120pixels_4_by_4deg_zs16_gnom_projection.npy')[:2000]

#############################################

###### Load the GSTools maps with Gaussian (as well as others) covariance model #######

#grf_example_s = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/grf_data_10000_maps_120_by_120_pixels_var_1.0_corrlen_5.0.npy')[:5000]
#erf_example_s = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/erf_data_10000_maps_120_by_120_pixels_var_1.0_corrlen_5.0_5.0.npy')
#grf_example_l = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/grf_data_10000_maps_120_by_120_pixels_var_1.0_corrlen_20.0.npy')[:5000]

#############################################

#### Load the Cosmogrid simulation maps with intrinsic alignment signal ####

#kappa_no_IA = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_plus_IA_NLA_A0.0_alpha_1.66_data_11600_maps_200_by_200pixels_20_by_20deg_4_des_zsbin_gnom_projection.npy')
#kappa_IA_Aamp = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_plus_IA_NLA_A0.36_alpha_1.66_data_11600_maps_200_by_200pixels_20_by_20deg_4_des_zsbin_gnom_projection.npy')
#kappa_IA_NLA1 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_plus_IA_NLA_A0.36_alpha_1.66_data_11600_maps_200_by_200pixels_20_by_20deg_4_des_zsbin_gnom_projection.npy')[:,2:,:,:]
#kappa_IA_NLA2 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_plus_IA_NLA_A0.79_alpha_4_data_11600_maps_200_by_200pixels_20_by_20deg_4_des_zsbin_gnom_projection.npy')[:,2:,:,:]

#### Load the Takahashi N-body maps and the flask log-normal maps with the same cosmology ####
#t17 = np.load('/e/ocean1/users/lgong/CCNN/takahashi/kappa_data_6264_maps_200_by_200pixels_20_by_20deg_zs16_gnom_projection.npy')[:,0,:,:]
#flask = np.load('/e/ocean1/users/lgong/CCNN/random_fields_data/maps_takahashi_cosmo/gaussian_lognormal_z10_z16/lognormal_data_12180_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy')[:6264,1,:,:]

"""Load snapshots following a spatial random distribution"""
def load_srf(num_snaps, size, srf_data):
    snaps = torch.zeros(num_snaps, 1, size, size, dtype=torch.float32)
    data = torch.tensor(srf_data, dtype=torch.float32)
    # Set the channel to random gaussian distribution
    snaps[:, 0, :, :] = data
    #snaps = data
    # Split into a train and val set
    num_train = int(0.9 * num_snaps)
    return snaps[:num_train], snaps[num_train:]

def load_multi_channel_data(num_snaps, srf_data):
    snaps = torch.tensor(srf_data, dtype=torch.float32)
    # Split into a train and val set
    num_train = int(0.9 * num_snaps)
    return snaps[:num_train], snaps[num_train:]

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

train_grf, val_grf = load_srf(5000, 200, grf_example)
train_log, val_log = load_srf(5000, 200, log_example)

#train_log_lambda1, val_log_lambda1 = load_srf(12180, 200, log_example_lambda1)
#train_log_lambda2, val_log_lambda2 = load_srf(12180, 200, log_example_lambda2)

#train_mv00, val_mv00 = load_srf(9999, 128, mv00_example)
#train_mv01, val_mv01 = load_srf(9999, 128, mv01_example)

#train_grf_s, val_grf_s = load_srf(5000, 120, grf_example_s)
#train_erf_s, val_erf_s = load_srf(10000, 120, erf_example_s)
#train_grf_l, val_grf_l = load_srf(5000, 120, grf_example_l)

#train_no_IA, val_no_IA = load_srf(11600, 200, kappa_no_IA)
#train_IA_A4, val_IA_A4 = load_srf(11600, 200, kappa_IA_A4)
#train_no_IA, val_no_IA = load_multi_channel_data(11600, kappa_no_IA)
#train_IA_Aamp, val_IA_Aamp = load_multi_channel_data(11600, kappa_IA_Aamp)

#train_t17, val_t17 = load_srf(6264, 200, t17)
#train_flask, val_flask = load_srf(6264, 200, flask)

"""
# data transformation functions
def _vflip(arr):
    return torch.flip(arr, [-2])

def _hflip(arr):
    return torch.flip(arr, [-1])

def _rot(arr, k):
    return torch.rot90(arr, k, [-2, -1])

## Modules used for performing symmetrization

class SymSlice(nn.Module):
    def __init__(self):
        super().__init__()

    #Given an array of shape [B, ..., W, H], produces an array of shape [B, 8, ..., W, H]
    of all symmetry transformations of the input, then folds the symmetry-equivalent versions
    into the batch dimension to make an array of shape [B * 8, ..., W, H]
          
    def forward(self, x):
        batch_size = x.shape[0]
        sym_x = torch.empty(batch_size, 8, *x.shape[1:]).to(x)
        sym_x[:, 0] = x
        sym_x[:, 1] = _hflip(x)
        sym_x[:, 2] = _vflip(x)
        sym_x[:, 3] = _rot(x, 1)
        sym_x[:, 4] = _rot(x, 2)
        sym_x[:, 5] = _rot(x, 3)
        sym_x[:, 6] = _rot(_hflip(x), 1)
        sym_x[:, 7] = _rot(_vflip(x), 1)
        x = sym_x.reshape(batch_size * 8, *x.shape[1:])
        return x
    

    # rotation over k*90 degree k belongs to {0,1,2,3}
    def forward(self, x):
        batch_size = x.shape[0]
        sym_x = torch.empty(batch_size, 4, *x.shape[1:]).to(x)
        sym_x[:, 0] = x
        sym_x[:, 1] = _rot(x, 1)
        sym_x[:, 2] = _rot(x, 2)
        sym_x[:, 3] = _rot(x, 3)
        x = sym_x.reshape(batch_size * 4, *x.shape[1:])
        return x

class SymPool(nn.Module):
    
    def __init__(self):
        super().__init__()
    
     
    #Performs inverse symmetry operations to bring feature maps back in line, then
    averages activations pixelwise. Must operate on a feature map that has been through
    SymSlice.
    
    def __init__(self, reduction='mean'):
        if reduction == 'mean':
            self.reduction = lambda x: torch.sum(x, dim=1)
        elif reduction == 'max':
            self.reduction = lambda x: torch.max(x, dim=1)[0]
        else:
            raise ValueError("Unknown reduction method passed to SymPool")
        super().__init__()
    
    def forward(self, x):
        inv_x = torch.empty(x.shape[0] // 8, 8, *x.shape[1:], device=x.device)
        x = x.reshape(x.shape[0] // 8, 8, *x.shape[1:])

        inv_x[:, 0] = x[:, 0]
        inv_x[:, 1] = _hflip(x[:, 1])
        inv_x[:, 2] = _vflip(x[:, 2])
        inv_x[:, 3] = _rot(x[:, 3], -1)
        inv_x[:, 4] = _rot(x[:, 4], -2)
        inv_x[:, 5] = _rot(x[:, 5], -3)
        inv_x[:, 6] = _hflip(_rot(x[:, 6], -1))
        inv_x[:, 7] = _vflip(_rot(x[:, 7], -1))
        return torch.sum(inv_x, dim=1)/8

    # inverse rotation over k*90 degree k belongs to {0,1,2,3}
    def forward(self, x):
        inv_x = torch.empty(x.shape[0] // 4, 4, *x.shape[1:], device=x.device)
        x = x.reshape(x.shape[0] // 4, 4, *x.shape[1:])

        inv_x[:, 0] = x[:, 0]
        inv_x[:, 1] = _rot(x[:, 1], -1)
        inv_x[:, 2] = _rot(x[:, 2], -2)
        inv_x[:, 3] = _rot(x[:, 3], -3)
        return torch.sum(inv_x, dim=1)/4
"""

# Set up the continuous planar rotation group SO(2)
bandlimit = 6
gspace = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=bandlimit)
#feat_type_in = escnn.nn.FieldType(gspace, [gspace.irrep(0)])
#feat_type_out = escnn.nn.FieldType(gspace, [gspace.irrep(0)])

""" Implements the non-linear convolution operation using the recursive formula."""
class NonlinearConv(nn.Module):
    def __init__(self, trial, in_channels, num_filts, filter_size, order):
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
    def __init__(self, trial):
        super().__init__()

    def forward(self, x):
        return x

""" Does a simple average over the last two dimensions of the input."""
class Mean2d(nn.Module):
    def __init__(self, trial):
        super().__init__()
        
    def forward(self, x):
        return x.mean((-2, -1))

""" Concatenate the one-column output from the logistic regression layer with a column of 0 to enable cross entropy loss calculation"""
class probability_concatenation(nn.Module):
    def __init__(self, trial):
        super().__init__()

    def forward(self, x):
        return torch.cat((x, torch.zeros_like(x)), dim=-1)

""" 
The full CCNN is Input -> NonlinearConv -> Spatial Avg -> BatchNorm -> Logistic. The way it is constructed here assumes a binary classification task, but this can be extended to the multiclass case.

Here the slice and pool variable indicate whether to carry out the symmetry slicing and pooling operations or not
"""
class CCNN(nn.Module):
    def __init__(self, trial, in_channels, num_filts, filter_size, order, absbeta=False, slice=False, pool=False):
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
        self.corr_conv = NonlinearConv(trial, self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d(trial)
        self.bn = nn.BatchNorm1d(self.num_filts * self.order, affine=True, eps=3.185289e-35, dtype=torch.float32)
        self.logistic_fc = nn.Linear(self.num_filts * self.order, 1, dtype=torch.float32)
        # If absbeta is set, force the beta_alpha^(n) parameters to be strictly positive
        # The effect of this is that the model can only pick up features which "give evidence for" class A
        self.concat = probability_concatenation(trial)
        
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

###############################################

"""Performs one training epoch, feeding an entire DataLoader through a model,
   performing gradient descent and keeping track of the average loss and accuracy."""
def train(model, optimizer, gamma, device='cuda'):
    
    model.train()
    for inputs, labels in train_loader:
        # Zero out gradients from the previous iteration
        optimizer.zero_grad()
        
        # Send the inputs and outputs to the GPU (if using one)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Feed the input through the network, measure the loss
        pred = model(inputs)
        loss = loss_criterion(pred, labels)
        
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
        
"""Performs one validation epoch, feeding an entire DataLoader through a model, and
   calculating the average loss and accuracy."""
def val(model, device='cuda'):
    
    model.eval()
    accuracy_total = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            pred = model(inputs)
            pred_classes = pred.argmax(dim=1).cpu()
            labels = labels.cpu()
            accuracy_total += torch.count_nonzero(pred_classes == labels).item() / len(pred_classes)
            
    avg_acc = accuracy_total / len(val_loader)
    return avg_acc


def objective(trial):
    """
    Objective function to be optimized by Optuna.

    Hyperparameters chosen to be optimized: optimizer, learning rate,
    filter_size, order of correlation estimators, regularization strength gamma

    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """

    # Define range of values to be tested for the hyperparameters
    #filter_size = trial.suggest_int("filter_size", 11, 55, 4) # kernel size of the convolution
    #corr_order = trial.suggest_int("corr_order", 3, 8) # up to which order of correlation we measure
    gamma = trial.suggest_float("regularization_gamma", 1e-3, 1e1, log=True) # regularization strength gamma
    lr_scheduler_ratio = trial.suggest_float("lr_scheduler_ratio", 1e-2, 1.0, log=True) # learning rate scheduler ratio
    #batchnorm_eps = trial.suggest_float("batchnorm_eps", 1e-100, 1e-7, log=True) # batch normalization stability factor 
    
    # Pre-defined hyperparameters
    filter_size = 31
    num_in_channels = 1
    num_filters = 1
    corr_order = 4
    n_epochs = 100                         # Number of training epochs
    
    # Generate the model
    #model = CCNN(trial, num_in_channels, num_filters, filter_size, corr_order, batchnorm_eps).to(device)
    model = CCNN(trial, num_in_channels, num_filters, filter_size, corr_order).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-4, 1.0, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    # A learning schedule which will ramp down the learning rate as we train
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_scheduler_ratio)

    # Training of the model
    for epoch in range(n_epochs):
        train(model, optimizer, gamma)  # Train the model
        accuracy = val(model)   # Evaluate the model
        # Step the learning rate schedule
        lr_scheduler.step()

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

start = time.time()

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Optimization study for a PyTorch CNN with Optuna
    # -------------------------------------------------------------------------

    # Use cuda if available for faster computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    
    # --- Parameters ----------------------------------------------------------
    number_of_trials = 20             # Number of Optuna trials
    
    # Make runs repeatable
    random_seed = 1750
    torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
    torch.manual_seed(random_seed)

    # Load the training and validation data set
    train_loader, val_loader = make_datasets(train_grf, train_log, val_grf, val_log)
    #train_loader, val_loader = make_datasets(train_grf_s, train_grf_l, val_grf_s, val_grf_l)
    
    # Create the loss criterion
    loss_criterion = nn.CrossEntropyLoss()
    
    #with open('../study_saves/ccnn_study_massivenus_1_channel_1_filter_0.0ev_vs_0.1ev_zs10_3.5_deg_128_pixels_order4.pickle', 'rb') as f:
    #    study = pickle.load(f)
    
    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="maximize") # for a loaded and continue-to-sample study object, one needs to comment this line out
    study.optimize(objective, n_trials=number_of_trials)
    
    #with open('../../study_saves/toy_models/ccnn_study_gstools_1_channel_1_filter_31pix_gaussiancov_5e3maps_lambda5_vs_20_120_pixels_order4_escnn.pickle', 'wb') as f:
    with open('../../study_saves/toy_models/ccnn_study_flask_1_channel_1_filter_31pix_t17_5e3maps_gaussian_vs_lognormal_zs16_200_pixels_order4_escnn.pickle', 'wb') as f:
        pickle.dump(study, f)
    
    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

end = time.time()
# Display the study statistics
print("\nStudy statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

trial = study.best_trial
print("Best trial:")
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
# Save results to csv file
df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
df = df.drop('state', axis=1)                 # Exclude state column
df = df.sort_values('value')                  # Sort based on accuracy

#df.to_csv('../../optuna_results_gstools/optuna_results_1_channel_1_filter_31pix_gaussiancov_5e3maps_lambda5_vs_20_120by120_pixels_order4_escnn_toy_models.csv', index=False) # Save to csv file
df.to_csv('../../optuna_results_gaussian_lognormal_t17_z16_only/optuna_results_1_channel_1_filter_31pix_t17_5e3maps_gaussian_vs_lognormal_200by200_pixels_order4_escnn_toy_models.csv', index=False) # Save to csv file

# Find the most important hyperparameters
most_important_parameters = optuna.importance.get_param_importances(study, target=None)

# Display the most important hyperparameters
print('\nMost important hyperparameters:')
for key, value in most_important_parameters.items():
    print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))


print("The time for 20 trial and each with 100 epochs is %s"%(end-start))