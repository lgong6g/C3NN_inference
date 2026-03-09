import sys
sys.path.append('./src')
from load_data import load_data_wide, load_data_full, load_data_narrow
#from load_data import load_data_wide_noaverage
import optuna
import pickle
import cloudpickle as cldpk
import numpy as np
import torch
import random
import time, datetime
import os
from main import *
from hyperparameters import hparams
from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn
from torch import from_numpy
import argparse

parser = argparse.ArgumentParser(description="My program description")
parser.add_argument("-t", "--test", action="store_true", help="Run in test mode")
args = parser.parse_args()
if args.test:
    print('Running in test mode')

# Make runs repeatable
random_seed = 1750 # that is the seed we have been used so far
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.enabled = True  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# choose database sqlite or pickle
database = 'pickle'
#database = 'sqlite'

"""
# load pretrained model
with open('/e/ocean1/users/lgong/C3NN_inference/c3nn-sbi_models/c3nn_tuned_model_param_3D_40filter_pix37_4order_nozscorex_bn_w_std_nocompression/trial_0/density_estimator.pickle', 'rb') as f:
    density_estimator = cldpk.load(f)

pretrained_c3nn = density_estimator._embedding_net
# Use the pretrained model filter weights as the initialization for the hybrid new model
# One needs to select the corresponding pixels indices for the new model
prefix_filter_weights = pretrained_c3nn.corr_conv.conv.expand_parameters()[0].abs()[:, :, 14:23, 14:23]
"""
# load data
print('loading data...')
if hparams.model == 'hybrid':
    training_parameters, training_data = load_data_hybrid(hparams)
else:
    training_parameters, training_data = load_data_wide(hparams)
    training_parameters_narrow, training_data_narrow = load_data_narrow(hparams)
    training_parameters = torch.cat((training_parameters,training_parameters_narrow))
    training_data = torch.cat((training_data,training_data_narrow))
    #training_parameters, training_data = training_parameters_narrow[50:], training_data_narrow[50:]
if args.test:
    training_parameters, training_data = training_parameters[:10], training_data[:10]

# Objective function to minimize
def objective(trial):

    # Hyperparameters to optimize
    flow_learning_rate = trial.suggest_float("flow_learning_rate", 1e-5, 1e-2, log=True)
    flow_num_transforms = trial.suggest_int("flow_num_transforms", 2, 8)
    #flow_num_transforms = trial.suggest_int("flow_num_transforms", 7, 14)
    flow_hidden_features = trial.suggest_categorical("flow_hidden_features", [32, 64, 128, 256, 512])
    #flow_hidden_features = trial.suggest_categorical("flow_hidden_features", [64, 128, 256, 512])
    num_blocks = trial.suggest_int("num_blocks", 2, 6)
    #num_blocks = trial.suggest_int("num_blocks", 5, 8)
    #flow_training_batch_size = trial.suggest_categorical("flow_training_batch_size", [16, 32, 64, 128])
    flow_training_batch_size = trial.suggest_categorical("flow_training_batch_size", [16])
    #num_filts = trial.suggest_int("num_filts", 1, 5) 
    filter_size = trial.suggest_categorical("filter_size", [37])
    #filter_size = trial.suggest_int("filter_size", 23, 47, step=2)

    # Some verbose
    print('\nTrial number: {}'.format(trial.number))
    print('flow_learning_rate:  {}'.format(flow_learning_rate))
    print('flow_num_transforms:  {}'.format(flow_num_transforms))
    print('flow_hidden_features:  {}'.format(flow_hidden_features))
    print('flow_num_blocks:  {}'.format(num_blocks))
    print('flow_training_batch_size:  {}'.format(flow_training_batch_size))
    #print('num_filts:  {}'.format(num_filts))
    print('filter_size:  {}'.format(filter_size))

    # Hyperparameters to be optimized
    hparams.flow_learning_rate = flow_learning_rate
    hparams.flow_num_transforms = flow_num_transforms
    hparams.flow_hidden_features = flow_hidden_features
    hparams.num_blocks = num_blocks
    hparams.flow_training_batch_size = flow_training_batch_size
    #hparams.num_filts = num_filts
    hparams.filter_size = filter_size

    # instatiate model, optimizer and loss function
    # initialize the new model filter weights from the pre-trained model
    #print('instantiating model...')
    #c3nn = C3NN_w_std(hparams).to(device='cuda')
    #c3nn = C3NN_Nptonly_Nptbn(hparams, prefix_filter_weights).to(device='cuda')
    #embedding_c3nn = CombinedC3NN_Nptbn(pretrained_c3nn, c3nn)

    # Run main routine
    val_prob, density_estimator, inference = main(hparams, training_parameters, training_data, verbose = True, test= args.test)

    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()

    return val_prob


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    #### Hypertuning using sqlite or pickle data base ####
    # Optuna parameters
    n_trials = 1

    if args.test:
        sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=5) # Use this line for loading the existing study and continue the tuning
        study = optuna.create_study(direction='maximize', sampler=sampler) # This line should be commmented out if loading the existing study
    elif database == 'pickle':

        # Load the saved study object and continue to tune hyperparameters
        try:
            with open('Models/optuna_scan_filter25_pix37_order2_3dim_param_nozscorex_bn_w_std_no_c1.pickle', 'rb') as f:
                study = pickle.load(f)
        except:
            print('No existing study found. Creating new study.')
            # Define sampler and start optimization
            #sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=10)
            sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=5) # Use this line for loading the existing study and continue the tuning
            study = optuna.create_study(direction='maximize', sampler=sampler) # This line should be commmented out if loading the existing study

    elif database == 'sqlite' and not args.test:
        storage = "sqlite:///"+os.getcwd()+"/optuna_scan_filter3_order4_deterministic_8params"
        study_name = "optuna_scan_filter3_order4_deterministic_8params"
        sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=5) # Use this line for loading the existing study and continue the tuning
        study = optuna.create_study(direction='maximize', study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)

    else:
        raise NotImplementedError("Database type not implemented.")


    
    study.optimize(objective, n_trials, gc_after_trial=True)

    if database == 'pickle' and args.test:
        with open('Models/testing_optuna.pickle', 'wb') as f:
            pickle.dump(study, f)
    elif database == 'pickle':
        with open('Models/hybrid.pickle', 'wb') as f:
            pickle.dump(study, f)
        
    # Print info for best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
