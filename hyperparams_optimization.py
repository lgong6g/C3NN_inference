import sys
sys.path.append('./src')
from load_data import load_data
import optuna
import pickle
import numpy as np
import torch
import random
import time, datetime
import os
from main import *
from hyperparameters import hparams


# Make runs repeatable
random_seed = 1750
torch.manual_seed(random_seed)
#torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms

# choose database sqlite or pickle
#database = 'pickle'
database = 'sqlite'

# load data
print('loading data...')
training_parameters, training_data = load_data(hparams)

# Objective function to minimize
def objective(trial):

    # Hyperparameters to optimize
    flow_learning_rate = trial.suggest_float("flow_learning_rate", 1e-5, 1e-2, log=True)
    flow_num_transforms = trial.suggest_int("flow_num_transforms", 2, 8)
    flow_hidden_features = trial.suggest_categorical("flow_hidden_features", [32, 64, 128, 256, 512])
    num_blocks = trial.suggest_int("num_blocks", 2, 6)
    #flow_training_batch_size = trial.suggest_categorical("flow_training_batch_size", [10, 20, 30, 40, 50])
    flow_training_batch_size = trial.suggest_categorical("flow_training_batch_size", [16, 32, 64, 128])
    #num_filts = trial.suggest_int("num_filts", 1, 5) 
    #filter_size = trial.suggest_categorical("filter_size", [15, 25, 35, 45, 55])
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

    # Hyperparameters to be optimized
    hparams.flow_learning_rate = flow_learning_rate
    hparams.flow_num_transforms = flow_num_transforms
    hparams.flow_hidden_features = flow_hidden_features
    hparams.num_blocks = num_blocks
    hparams.flow_training_batch_size = flow_training_batch_size
    #hparams.num_filts = num_filts
    hparams.filter_size = filter_size


    # Run main routine
    val_prob = main(hparams, training_parameters, training_data, verbose = True)

    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()

    return val_prob


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    #### Hypertuning using sqlite or pickle data base ####
    # Optuna parameters
    n_trials = 100

    if database == 'pickle':

        # Load the saved study object and continue to tune hyperparameters
        try:
            with open('optuna_test.pickle', 'rb') as f:
                study = pickle.load(f)
        except:
            print('No existing study found. Creating new study.')
            # Define sampler and start optimization
            #sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=10)
            sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=20) # Use this line for loading the existing study and continue the tuning
            study = optuna.create_study(direction='maximize', sampler=sampler) # This line should be commmented out if loading the existing study

    elif database == 'sqlite':
        storage = "sqlite:///"+os.getcwd()+"/optuna_testsqlite"
        study_name = "optuna_testsqlite"
        sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=20) # Use this line for loading the existing study and continue the tuning
        study = optuna.create_study(direction='maximize', study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)

    else:
        raise NotImplementedError("Database type not implemented.")


    
    study.optimize(objective, n_trials, gc_after_trial=True)

    if database == 'pickle':
        with open('optuna_test.pickle', 'wb') as f:
            pickle.dump(study, f)
        
    # Print info for best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
