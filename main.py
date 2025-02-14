import sys
sys.path.append('./src')
from hyperparameters import hparams
from C3NN import C3NN
from load_data import load_data
from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn
import numpy as np
import random
from torch import from_numpy
import torch
import time, datetime



def main(hparams, verbose=True):

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # load data
    training_parameters, training_data = load_data(hparams)

    # instatiate model, optimizer and loss function
    c3nn = C3NN(hparams)
    #c3nn = C3NN(in_channels=in_channels, num_filts=num_filts, filter_size=filter_size, order=order, absbeta=False).to(device='cuda')

    # Set up prior for sbi
    lower = np.array([0.1, 0.4])
    upper = np.array([0.5, 1.4])
    prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device=hparams.device)

    #set up sbi architecture
    density_estimator_custom = posterior_nn(model="maf", hidden_features=hparams.flow_hidden_features, num_transforms=hparams.flow_num_transforms, device=hparams.device, embedding_net=c3nn)
    inference = SNPE_C(prior, device=hparams.device, density_estimator=density_estimator_custom)

    # joint training
    density_estimator = inference.append_simulations(training_parameters[:10], training_data[:10]).train(learning_rate=hparams.learning_rate,training_batch_size=hparams.flow_training_batch_size,max_num_epochs=2)

    val_loss = inference.summary['best_validation_log_prob']

    # main training routine; trains until early stopping is reached
    #train_loss, val_loss = training_routine(model, optimizer, loss_fn, train_loader, val_loader, hparams, device=device)


    if verbose:
        print(val_loss)

    return(val_loss)




if __name__ == "__main__":

    time_ini = time.time()

    main(hparams)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))