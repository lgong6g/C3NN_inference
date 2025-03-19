import optuna
import pickle
import numpy as np
import random
import time, datetime
import os

import torch
import torch.nn as nn
from torch import from_numpy
import torch.nn.functional as F

import escnn.gspaces
import escnn.nn

from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn

# Make runs repeatable
random_seed = 1750
torch.manual_seed(random_seed)
torch.cuda.empty_cache()
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms

training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
#training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')

training_parameters = np.load('CosmoGrid_params.npz').files
param_names = ['Om', 'sigma8']
training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')

bandlimit = 6
gspace = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=bandlimit)

""" Implements the non-linear convolution operation using the recursive formula."""
class NonlinearConv(nn.Module):
    def __init__(self, trial, in_channels, num_filts, filter_size, order):
        super().__init__()
        self.order = order
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        
        # Initialize the escnn filter
        feat_type_in = escnn.nn.FieldType(gspace, self.in_channels*[gspace.trivial_repr])
        feat_type_out = escnn.nn.FieldType(gspace, self.num_filts*[gspace.trivial_repr])
        # THIS LAYER KILLS DEEPCOPY
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
    def __init__(self, trial):
        super().__init__()
        
    def forward(self, x):
        return x.mean((-2, -1))


""" The full CCNN is Input -> NonlinearConv -> Spatial Avg -> BatchNorm -> Logistic. The way it is constructed
    here assumes a binary classification task, but this can be extended to the multiclass case."""
class C3NN(nn.Module):
    def __init__(self, trial, in_channels, num_filts, filter_size, order, absbeta=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order
        self.absbeta = absbeta

        # the model architecture with escnn
        self.corr_conv = NonlinearConv(trial, self.in_channels, self.num_filts, self.filter_size, self.order)
        self.spatial_mean = Mean2d(trial)
        # I preserve the batch normalization layer here so that we can scale the compressed summary statistics to the same order of magnitude
        # This may help the numerical stability of inference in the 2nd part
        self.bn = nn.BatchNorm1d(self.num_filts * self.order, affine=True, eps=3.185289e-35, dtype=torch.float32)
        self.parallel = torch.vmap(self.forward_single, in_dims=0, out_dims=0)
    
    def forward_single(self, x, chop=True):    
        
        x = self.corr_conv(x)
        #x = self.conv(x)
        x = self.spatial_mean(x)
        if chop == False:
            x = self.bn(x)
        return x
    
    def forward(self, x):
        return torch.mean(self.parallel(x), 1)

class Inference:
    def __init__(self, trial, model, flow_hidden_features, flow_num_transforms, num_blocks, learning_rate, flow_training_batch_size, device='cuda'):
        self.model = model
        self.flow_hidden_features = flow_hidden_features
        self.flow_num_transforms = flow_num_transforms
        self.num_blocks = num_blocks
        self.learning_rate = learning_rate
        self.flow_training_batch_size = flow_training_batch_size
        self.device = device
        self.prior = self._set_prior()

    def _set_prior(self):
        lower = np.array([0.1, 0.4])
        upper = np.array([0.5, 1.4])
        return utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper), device=self.device)
    
    def validate(self, parameters, data, verbose=True):
        print('Instantiating SBI...')
        density_estimator_custom = posterior_nn(model="maf", hidden_features=self.flow_hidden_features, num_transforms=self.flow_num_transforms, num_blocks=self.num_blocks, device=self.device, embedding_net=self.model)
        
        inference = SNPE_C(self.prior, device=self.device, density_estimator=density_estimator_custom)
        density_estimator = inference.append_simulations(parameters, data).train(learning_rate=self.learning_rate, training_batch_size=self.flow_training_batch_size, max_num_epochs=5000)

        val_loss = inference.summary['best_validation_log_prob']
        
        if verbose:
            print(val_loss)
        
        return val_loss

def objective(trial):

    # Hyperparameters to optimize
    flow_learning_rate = trial.suggest_float("flow_learning_rate", 1e-5, 1e-2, log=True)
    flow_num_transforms = trial.suggest_int("flow_num_transforms", 2, 8)
    flow_hidden_features = trial.suggest_categorical("flow_hidden_features", [32, 64, 128, 256, 512])
    num_blocks = trial.suggest_int("num_blocks", 2, 6)
    flow_training_batch_size = trial.suggest_categorical("flow_training_batch_size", [8, 16, 32, 64, 128])
    #num_filts = trial.suggest_int("num_filts", 1, 5) 
    filter_size = trial.suggest_int("filter_size", 23, 47, step=2)

    # Some verbose
    print('\nTrial number: {}'.format(trial.number))
    print('flow_learning_rate:  {}'.format(flow_learning_rate))
    print('flow_num_transforms:  {}'.format(flow_num_transforms))
    print('flow_hidden_features:  {}'.format(flow_hidden_features))
    print('flow_num_blocks:  {}'.format(num_blocks))
    print('flow_training_batch_size:  {}'.format(flow_training_batch_size))
    #print('num_filts:  {}'.format(num_filts))
    print('filter_size:  {}'.format(filter_size))

    # Hyperparameters to be fixed
    in_channels = 4
    num_filts = 1
    order = 2

    c3nn = C3NN(trial, in_channels, num_filts, filter_size, order).to(device='cuda')
    flow = Inference(trial, c3nn, flow_hidden_features, flow_num_transforms, num_blocks, flow_learning_rate, flow_training_batch_size)
    
    # Run validation routine
    val_prob = flow.validate(training_parameters, training_data)

    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()

    return val_prob

if __name__ == "__main__":

    time_ini = time.time()


    # Optuna parameters
    #storage = "sqlite:///"+os.getcwd()+"/optuna_scan_filter1_order2_new_code"
    #storage = "optuna_scan"
    #study_name = "hparam_scan_new_code"
    n_trials = 1

    #sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=10)
    #study = optuna.create_study(direction='maximize', study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials, gc_after_trial=True)
    
    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
    
    with open('optuna_scan_filter1_order2_time_test.pickle', 'wb') as f:
        pickle.dump(study, f)

    # Print info for best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

    





