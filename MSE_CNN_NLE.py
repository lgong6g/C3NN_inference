import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import pandas as pd
import random
import cloudpickle as cpk
from chainconsumer import ChainConsumer

import torch
import torch.nn as nn

from sbi import __version__ as sbi_version
from sbi.inference import SNLE
from sbi.utils import BoxUniform
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn

from src.load_data import load_data_wide, load_data_narrow
from hyperparameters import hparams

from torch import from_numpy

# Set random seeds for reproducibility
RANDOM_SEED = 42

# Set Python random seed
random.seed(RANDOM_SEED)

# Set NumPy random seed
np.random.seed(RANDOM_SEED)

# Set PyTorch random seed
torch.manual_seed(RANDOM_SEED)

# Set CUDA random seed if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # Make CUDA operations deterministic (may impact performance)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

# Set environment variable for additional reproducibility
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

print(f"SBI version: {sbi_version}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")
print(f"Random seed set to: {RANDOM_SEED}")

#training_parameters, training_data = load_data_wide(hparams)
#training_parameters_narrow, training_data_narrow = load_data_narrow(hparams)

#training_parameters_mean = training_parameters.mean(0)
#training_parameters_std = training_parameters.std(0)
#training_parameters_stand = (training_parameters - training_parameters_mean) / training_parameters_std
#training_parameters_stand_narrow = (training_parameters_narrow - training_parameters_mean) / training_parameters_std
'''

from torch.utils.data import TensorDataset, DataLoader
#train_dataset = TensorDataset(training_parameters_stand, training_data)
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
#train_dataset_narrow = TensorDataset(training_parameters_stand_narrow, training_data_narrow)
#train_loader_narrow = DataLoader(train_dataset_narrow, batch_size=64, shuffle=False)

import cloudpickle as cpk
with open(f'/project/ag-weller/Kai.Lehman/C3NN_Models/MSE_Models/3convs_bandwidth2_p63.cpk', 'rb') as f:
    model = cpk.load(f)


with open(f'/project/ag-weller/Kai.Lehman/C3NN_Models/MSE_Models/3convs_bandwidth2_p63_wide_summary.cpk', 'rb') as f:
    y_pred = cpk.load(f)
with open(f'/project/ag-weller/Kai.Lehman/C3NN_Models/MSE_Models/3convs_bandwidth2_p63_narrow_summary.cpk', 'rb') as f:
    y_pred_narrow = cpk.load(f)

print('everything loaded')


y_comb = torch.cat((y_pred.detach(),y_pred_narrow.detach()))
training_parameters_comb = torch.cat((training_parameters,training_parameters_narrow))
print(y_comb.shape,training_parameters_comb.shape)


lower = np.array([0.1, 0.4, -2.0, 0.87, 0.03, 64.0, 12.0, -2.0])
upper = np.array([0.5, 1.4, -0.33, 1.07, 0.06, 82.0, 15.0, 2.0])
prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device=hparams.device)

# Define the MAF density estimator
density_estimator = posterior_nn(
    model='maf',
    num_transforms=4,         # number of MADEs
    num_blocks=3,             # number of layers in each MADE
    hidden_features=128,      # number of hidden units per layer
)

# Set up SNPE_C with the MAF as density estimator and the compressor as embedding network
#inference = SNPE_C(prior=prior, show_progress_bars=True, density_estimator=density_estimator, device=device)
inference = SNLE(prior=prior, show_progress_bars=True, density_estimator=density_estimator, device=device)
# THIS IS SNLE_A, IS THIS CORRECT TO USE?

_ = inference.append_simulations(training_parameters, y_pred.detach()).train(
    training_batch_size=32, 
    learning_rate=1e-4, 
    stop_after_epochs=20,
    max_num_epochs=1024,
    show_train_summary=True
)

#likelihood_estimator = inference.get_density_estimator()
'''

with open(f'/project/ag-weller/Kai.Lehman/C3NN_Models/MSE_Models/3convs_bandwidth2_p63_NLE_inference.cpk', 'rb') as f:
    inference = cpk.load(f)


posterior = inference.build_posterior()


#testing_data = np.load("/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_fiducial_cosmology_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy")
#testing_data = torch.unsqueeze(torch.tensor(testing_data, dtype=torch.float32).to(device='cuda'),0)
#testing_summ_stand = model(testing_data)
#testing_summ = testing_summ_stand *training_parameters_std + training_parameters_mean
with open(f"/project/ag-weller/Kai.Lehman/C3NN_Models/MSE_Models/3convs_bandwidth2_p63_test_summary.cpk", 'rb') as f:
    testing_summ = cpk.load(f)
#print(testing_data.shape)
print(testing_summ.shape)
print(testing_summ[0])
samples = posterior.sample((10000,), x=testing_summ[0]).cpu().numpy() #100000
#samples = remove_invalid_points(samples)
print(samples.shape)

sample_c2_nozscorex_bn_25filter = np.load('/project/ag-weller/Kai.Lehman/C3NN_Models/test_chains/25filters_2order_nozscorex_bn.npy')
sample_c3_nozscorex_bn_25filter = np.load('/project/ag-weller/Kai.Lehman/C3NN_Models/test_chains/25filters_3order_nozscorex_bn.npy')
sample_c4_nozscorex_bn_40filter = np.load('/project/ag-weller/Kai.Lehman/C3NN_Models/test_chains/40filters_4order_nozscorex_bn.npy')


c = ChainConsumer()

#post_df = pd.DataFrame(samples, columns=param_names_latex)
#truth_dict = dict(zip(param_names_latex, y_train[-1]))

c.add_chain(sample_c2_nozscorex_bn_25filter, parameters=["$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$n_s$", r"$\Omega_b$", r"$H_0$", r"$M_c^0$", r"$\nu$"], name=r'$\{c_1, c_2\}$ '+'25 filters no standardization plus BatchNorm', shade=True, shade_alpha=0.1, color='purple', linestyle='-', linewidth=2.0, kde=False)
c.add_chain(sample_c3_nozscorex_bn_25filter, parameters=["$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$n_s$", r"$\Omega_b$", r"$H_0$", r"$M_c^0$", r"$\nu$"], name=r'$\{c_1, c_2, c_3\}$ '+'25 filters no standardization plus BatchNorm', shade=True, shade_alpha=0.3, color='b', linestyle='-', linewidth=2.0, kde=False)
c.add_chain(sample_c4_nozscorex_bn_40filter, parameters=["$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$n_s$", r"$\Omega_b$", r"$H_0$", r"$M_c^0$", r"$\nu$"], name=r'$\{c_1, \ldots, c_4\}$ '+'40 filters no standardization plus BatchNorm', shade=True, shade_alpha=0.7, color='green', linestyle='-', linewidth=2.0, kde=False)
c.add_chain(samples, parameters=["$\Omega_m$", r"$\sigma_8$", r"$w_0$", r"$n_s$", r"$\Omega_b$", r"$H_0$", r"$M_c^0$", r"$\nu$"], name=r'$\{c_1, c_2\}$ '+'CNN MSE compression, NLE on both grids', shade=True, shade_alpha=0.1, color='orange', linestyle='-', linewidth=2.0, kde=False)


fig = c.plotter.plot(figsize=(12, 12), truth=[0.26, 0.84, -1, 0.9649, 0.0493, 67.3, 13.82, 0.0], legend=True)

plt.savefig('Plots/MSE_CNN_NLE.png')


print(samples.std(axis=0))
print(c.analysis.get_summary())
