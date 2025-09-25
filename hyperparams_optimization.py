#import numexpr
#print("Max threads allowed:", numexpr.detect_number_of_cores())
#print("Threads actually used:", numexpr.get_num_threads())
import sys
sys.path.append('./src')
from load_data import load_data_wide
#from load_data import load_data_full
import optuna
import pickle
import cloudpickle as cldpk
import numpy as np
import torch
import random
import time, datetime
import os
#from main import *
from hyperparameters import hparams
from KDE_prior_estimation import KDEDistribution
from C3NN import C3NN
from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn
from torch import from_numpy

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

# load data
print('loading data...')
training_parameters, training_data = load_data_wide(hparams)
#training_parameters, training_data = load_data_full(hparams)

# Objective function to minimize
def objective(trial):

    # Hyperparameters to optimize
    flow_learning_rate = trial.suggest_float("flow_learning_rate", 1e-5, 1e-2, log=True)
    flow_num_transforms = trial.suggest_int("flow_num_transforms", 2, 8)
    flow_hidden_features = trial.suggest_categorical("flow_hidden_features", [32, 64, 128, 256, 512])
    num_blocks = trial.suggest_int("num_blocks", 2, 6)
    #flow_training_batch_size = trial.suggest_categorical("flow_training_batch_size", [10, 20, 30, 40, 50])
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
    print('instantiating model...')
    c3nn = C3NN(hparams).to(device='cuda')
    #c3nn = C3NN(in_channels=in_channels, num_filts=num_filts, filter_size=filter_size, order=order, absbeta=False).to(device='cuda')

    # Set up prior for sbi
    # The param here are Om, sigma8, w0, ns, Ob, H0, bary_Mc, bary_nu
    lower = np.array([0.1, 0.4, -2.0, 0.87, 0.03, 64.0, 12.0, -2.0])
    upper = np.array([0.5, 1.4, -0.33, 1.07, 0.06, 82.0, 15.0, 2.0])
    prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device=hparams.device)

    #set up sbi architecture
    density_estimator_custom = posterior_nn(model="maf", hidden_features=hparams.flow_hidden_features, num_transforms=hparams.flow_num_transforms, num_blocks=hparams.num_blocks, device=hparams.device, embedding_net=c3nn, z_score_x=None)
    inference = SNPE_C(prior, device=hparams.device, density_estimator=density_estimator_custom)

    # joint training
    density_estimator = inference.append_simulations(training_parameters, training_data).train(learning_rate=hparams.learning_rate, training_batch_size=hparams.flow_training_batch_size, max_num_epochs=300)

    val_loss = inference.summary['best_validation_log_prob']

    # main training routine; trains until early stopping is reached
    #train_loss, val_loss = training_routine(model, optimizer, loss_fn, train_loader, val_loader, hparams, device=device)
    
    print(val_loss)

    # Run main routine
    #val_prob, density_estimator, inference = main(hparams, training_parameters, training_data, verbose = True)

    #if torch.cuda.is_available():
    #    torch.cuda.empty_cache()
    model_dir = f"/e/ocean1/users/lgong/C3NN_inference/c3nn-sbi_models/c3nn_tuned_model_param_8D_40filter_pix37_4order_nozscorex_bn_perturbed2sets/trial_{trial.number}"
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "density_estimator.pickle"), "wb") as f:
        cldpk.dump(density_estimator, f)

    with open(os.path.join(model_dir, "inference.pickle"), "wb") as f:
        cldpk.dump(inference, f)

    return val_loss


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    #### Hypertuning using sqlite or pickle data base ####
    # Optuna parameters
    n_trials = 4

    if database == 'pickle':

        # Load the saved study object and continue to tune hyperparameters
        try:
            with open('Models/optuna_scan_filter40_pix37_order4_8dim_param_nozscorex_bn_perturbed2sets.pickle', 'rb') as f:
                study = pickle.load(f)
        except:
            print('No existing study found. Creating new study.')
            # Define sampler and start optimization
            #sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=10)
            sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=5) # Use this line for loading the existing study and continue the tuning
            study = optuna.create_study(direction='maximize', sampler=sampler) # This line should be commmented out if loading the existing study

    elif database == 'sqlite':
        storage = "sqlite:///"+os.getcwd()+"/optuna_scan_filter3_order4_deterministic_8params"
        study_name = "optuna_scan_filter3_order4_deterministic_8params"
        sampler = optuna.samplers.TPESampler(seed=1750, n_startup_trials=5) # Use this line for loading the existing study and continue the tuning
        study = optuna.create_study(direction='maximize', study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)

    else:
        raise NotImplementedError("Database type not implemented.")


    
    study.optimize(objective, n_trials, gc_after_trial=True)

    if database == 'pickle':
        with open('Models/optuna_scan_filter40_pix37_order4_8dim_param_nozscorex_bn_perturbed2sets.pickle', 'wb') as f:
            pickle.dump(study, f)
        
    # Print info for best trial
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))



    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
