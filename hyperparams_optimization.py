import optuna
import numpy as np
import torch
import random
import time, datetime
import os
from main import *
from hyperparameters import hparams

#torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)

# Make runs repeatable
random_seed = 1750
torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms
torch.manual_seed(random_seed)

# Objective function to minimize
def objective(trial):

    # Hyperparameters to optimize
    flow_learning_rate = trial.suggest_float("flow_learning_rate", 1e-6, 1e-3, log=True)
    flow_num_transforms = trial.suggest_int("flow_num_transforms", 2, 8)
    flow_hidden_features = trial.suggest_categorical("flow_hidden_features", [32, 64, 128, 256, 512])
    num_blocks = trial.suggest_int("num_blocks", 2, 6)
    flow_training_batch_size = trial.suggest_categorical("flow_training_batch_size", [5, 10, 20, 30, 40, 50])
    #num_filts = trial.suggest_int("num_filts", 1, 5) 
    #filter_size = trial.suggest_categorical("filter_size", [15, 25, 35, 45, 55])
    filter_size = trial.suggest_int("filter_size", 23, 47, 2)

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
    val_prob = main(hparams, verbose = True)

    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()

    return val_prob


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()


    # Optuna parameters
    storage = "sqlite:///"+os.getcwd()+"/optuna_scan_filter1_order4"
    #storage = "optuna_scan"
    study_name = "hparam_scan"
    n_trials = 50

    # Define sampler and start optimization
    #sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=10)
    sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=10)
    study = optuna.create_study(direction='maximize', study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials, gc_after_trial=True)

    # Print info for best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
