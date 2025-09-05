import sys
sys.path.append('./src')
from hyperparameters import hparams
from C3NN import C3NN
from load_data import load_data_wide
from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn
import numpy as np
import random
from torch import from_numpy
import torch
import time, datetime
import cloudpickle as cpk



def main(hparams, training_parameters, training_data, verbose=True):

    # Make runs repeatable
    random_seed = 1750
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = True  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # load data
    #print('loading data...')
    #training_parameters, training_data = load_data_wide(hparams)

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
    density_estimator_custom = posterior_nn(model="mdn", hidden_features=hparams.flow_hidden_features, num_transforms=hparams.flow_num_transforms, num_blocks=hparams.num_blocks, device=hparams.device, embedding_net=c3nn)
    inference = SNPE_C(prior, device=hparams.device, density_estimator=density_estimator_custom)

    # joint training
    density_estimator = inference.append_simulations(training_parameters, training_data).train(learning_rate=hparams.learning_rate, training_batch_size=hparams.flow_training_batch_size, max_num_epochs=300)

    val_loss = inference.summary['best_validation_log_prob']

    # main training routine; trains until early stopping is reached
    #train_loss, val_loss = training_routine(model, optimizer, loss_fn, train_loader, val_loader, hparams, device=device)

    if hparams.save:
        with open(f"Models/{hparams.name_model()}_inference.pkl", "wb") as handle:
            cpk.dump(inference, handle)
        with open(f"Models/{hparams.name_model()}_density_estimator.pkl", "wb") as handle:
            cpk.dump(density_estimator, handle)
        print('Model saved')
    else:
        print('Not saving')

    if verbose:
        print(val_loss)

    return val_loss, density_estimator, inference




if __name__ == "__main__":

    time_ini = time.time()

    training_parameters, training_data = load_data_wide(hparams)
    #training_parameters, training_data = load_data_narrow(hparams)
    main(hparams, training_parameters, training_data)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
