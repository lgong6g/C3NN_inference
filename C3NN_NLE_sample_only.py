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

from sbi.inference import likelihood_estimator_based_potential
import matplotlib.path as mpltPath
import emcee


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


lower = np.array([0.1, 0.4, -2.0])
upper = np.array([0.5, 1.4, -0.33])
prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device=hparams.device)


###############
# reading in previously trained NLE network and previously computed summary
############

model_dir = f"/project/ag-weller/Kai.Lehman/C3NN_Models/"
#with open(f"{model_dir}/nle_widenarrow_inference.pkl", "rb") as f:
with open(f"{model_dir}/nle_widenarrow_mdn_inference.pkl", "rb") as f:
    inference = cpk.load(f)

#with open(f"{model_dir}/nle_widenarrow_density_estimator.pkl", "rb") as f:
with open(f"{model_dir}/nle_widenarrow_mdn_density_estimator.pkl", "rb") as f:
    density_estimator = cpk.load(f)

posterior = inference.build_posterior()

#testing_summ = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/fiducial_summ.npy')
testing_summ = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/fiducial_summ_mdn.npy')


######
# sampling
########
boundary_Om_sig8 = np.array([
    [0.1, 1.4],
    [0.2, 1.4],
    [0.5, 0.65],
    [0.5, 0.4],
    [0.3, 0.4],
    [0.1, 0.9],])
boundary_Om_w0 = np.array([
    [.1, -1.1],
    [.5, -2.],
    [.5, -.33],
    [.1, -.33],])
validOS = mpltPath.Path(boundary_Om_sig8)
validOW = mpltPath.Path(boundary_Om_w0)
potential_fn, parameter_transform = likelihood_estimator_based_potential(density_estimator, prior, testing_summ[0]) # density_estimator is you trained flow, prior is just the sbi box_prior
def log_prob(args):
    pot = np.array(potential_fn(torch.Tensor(args).to(device)).tolist())
    #pot[~validOS.contains_points(args[:,[0,4]])] = -np.inf #change indices to Om, s8
    #pot[~validOW.contains_points(args[:,[0,5]])] = -np.inf ##change indices to Om, w0
    pot[~validOS.contains_points(args[:,[0,1]])] = -np.inf #change indices to Om, s8
    pot[~validOW.contains_points(args[:,[0,2]])] = -np.inf ##change indices to Om, w0
    return pot
#theta_0 =  CGPrior.sample(nwalkers).detach().cpu().numpy() #Just replace this with some random initial points
nwalkers = 16
ndim = 3
theta_0 = np.random.uniform([.1,.7,-1.1],[.4,.9,-.9],(nwalkers,3))
nsteps = 100000
discard = 10000
step = 3
nflows = 1 #???
tot_steps = nflows * nsteps * step + discard
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True)
state = sampler.run_mcmc(theta_0, tot_steps, progress=True)
samples = sampler.get_chain(discard=discard, thin=step, flat=True) 
#samples = remove_invalid_points(samples)
print(samples.shape)


########
# plotting
########

sample_c4 = np.load('/project/ag-weller/Kai.Lehman/C3NN_Models/test_chains/40filters_4order_3d_nozscorex_bn_w_std_nocompression.npy')

c = ChainConsumer()

c.add_chain(sample_c4, parameters=["$\Omega_m$", r"$\sigma_8$", r"$w_0$"], name=r'$\{c_1, \ldots, c_4\}$ ', shade=True, shade_alpha=0.7, color='green', linestyle='-', linewidth=2.0, kde=False)
c.add_chain(samples, parameters=["$\Omega_m$", r"$\sigma_8$", r"$w_0$"], name=r'$\{c_1, ..., c_4\}$ '+'C3NN MAF compression, NLE on both grids', shade=True, shade_alpha=0.1, color='orange', linestyle='-', linewidth=2.0, kde=False)


fig = c.plotter.plot(figsize=(12, 12), truth=[0.26, 0.84, -1], legend=True)

#plt.savefig('Plots/C3NN_NLE_newsampling.png')
plt.savefig('Plots/C3NN_NLE_newsampling_mdn.png')


print(samples.std(axis=0))
print(c.analysis.get_summary())
