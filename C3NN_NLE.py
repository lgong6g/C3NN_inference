import sys
sys.path.append('./src')
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
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn

from src.load_data import load_data_wide, load_data_narrow
from hyperparameters import hparams

from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader

torch.cuda.empty_cache


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

training_parameters, training_data = load_data_wide(hparams)
training_parameters_narrow, training_data_narrow = load_data_narrow(hparams)
training_parameters = torch.cat((training_parameters,training_parameters_narrow))
training_data = torch.cat((training_data,training_data_narrow))

print('creating summary')

##########
# reading in VMIM trained summary network and creatings summaries for the simulations
########

#with open(f'/project/ag-weller/Kai.Lehman/C3NN_Models/order_4_filters_40_size_37_lr_1.00e-03_features_32_blocks_2_transforms_7_batchsize_16_widenarrow_density_estimator.pkl', 'rb') as f:
with open(f'/project/ag-weller/Kai.Lehman/C3NN_Models/order_4_filters_40_size_37_lr_1.00e-03_features_32_blocks_2_transforms_7_batchsize_16_widenarrow_mdn_density_estimator.pkl', 'rb') as f:
    density_estimator = cpk.load(f)

train_dataset = TensorDataset(training_parameters, training_data)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False,drop_last=False)

summ = np.empty((2500,2*4*40))
for i, (param, data) in enumerate(train_loader):
    summ[4*i:4*(i+1)] = density_estimator._embedding_net(data).detach().cpu()


#############
# creating summary for test simulation for later
###########

testing_data = np.load("/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_fiducial_cosmology_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy")
testing_data = torch.tensor(testing_data, dtype=torch.float32).to(device='cuda')

testing_summ = density_estimator._embedding_net(testing_data.unsqueeze(0)).detach().cpu()

#print(testing_data.shape)
print(testing_summ.shape)
print(testing_summ[0])

#np.save("/project/ag-weller/Kai.Lehman/C3NN_data/fiducial_summ.npy",testing_summ)
np.save("/project/ag-weller/Kai.Lehman/C3NN_data/fiducial_summ_mdn.npy",testing_summ)

print('trianing NLE')


#############
# training NLE with the summary and saving NLE network
###########

del train_dataset
del density_estimator
torch.cuda.empty_cache()

lower = np.array([0.1, 0.4, -2.0])
upper = np.array([0.5, 1.4, -0.33])
prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device=hparams.device)

# Define the MAF density estimator
density_estimator = likelihood_nn(
    model='maf',
    num_transforms=5,         # number of MADEs
    num_blocks=3,             # number of layers in each MADE
    hidden_features=128,      # number of hidden units per layer
)

#inference = SNPE_C(prior=prior, show_progress_bars=True, density_estimator=density_estimator, device=device)
inference = SNLE(prior=prior, show_progress_bars=True, density_estimator=density_estimator, device=device)

summ = torch.tensor(summ, dtype=torch.float32)



density_estimator = inference.append_simulations(training_parameters, summ).train(
    training_batch_size=32, 
    learning_rate=1e-4, 
    stop_after_epochs=20,
    max_num_epochs=1024,
    show_train_summary=True
)

#likelihood_estimator = inference.get_density_estimator()
model_dir = f"/project/ag-weller/Kai.Lehman/C3NN_Models/"
#with open(f"{model_dir}/nle_widenarrow_inference.pkl", "wb") as handle:
with open(f"{model_dir}/nle_widenarrow_mdn_inference.pkl", "wb") as handle:
    cpk.dump(inference, handle)
#with open(f"{model_dir}/nle_widenarrow_density_estimator.pkl", "wb") as handle:
with open(f"{model_dir}/nle_widenarrow_mdn_density_estimator.pkl", "wb") as handle:
    cpk.dump(density_estimator, handle)

posterior = inference.build_posterior()
