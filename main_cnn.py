import os
import sys
sys.path.append('./src')
import pickle
from hyperparameters import hparams, hparams_cnn
from CNN import CNNEmbedding
from CNN_resnet import ResidualBlock, CNNResEmbedding
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



def main(hparams, hparams_cnn, training_parameters, training_data, verbose=True):

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
    #cnn = CNNEmbedding(hparams_cnn).to(device='cuda')
    cnn = CNNResEmbedding().to(device='cuda')
    
    # Set up prior for sbi
    # The param here are Om, sigma8, w0, ns, Ob, H0, bary_Mc, bary_nu
    lower = np.array([0.1, 0.4, -2.0, 0.87, 0.03, 64.0, 12.0, -2.0])
    upper = np.array([0.5, 1.4, -0.33, 1.07, 0.06, 82.0, 15.0, 2.0])
    prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device=hparams.device)

    #set up sbi architecture
    density_estimator_custom = posterior_nn(model="maf", hidden_features=hparams.flow_hidden_features, num_transforms=hparams.flow_num_transforms, num_blocks=hparams.num_blocks, device=hparams.device, embedding_net=cnn, z_score_x=None)
    inference = SNPE_C(prior, device=hparams.device, density_estimator=density_estimator_custom)

    # joint training
    density_estimator = inference.append_simulations(training_parameters, training_data).train(learning_rate=hparams.learning_rate, training_batch_size=hparams.flow_training_batch_size, max_num_epochs=300)

    val_loss = inference.summary['best_validation_log_prob']

    # main training routine; trains until early stopping is reached
    #train_loss, val_loss = training_routine(model, optimizer, loss_fn, train_loader, val_loader, hparams, device=device)

    if hparams_cnn.save:
        #model_dir = f"/e/ocean1/users/lgong/C3NN_inference/cnn-sbi_models/model_{hparams_cnn.name_model()}"
        model_dir = f"/e/ocean1/users/lgong/C3NN_inference/cnn-sbi_models/model_cnn_resnet"
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "density_estimator.pickle"), "wb") as f:
            cpk.dump(density_estimator, f)

        with open(os.path.join(model_dir, "inference.pickle"), "wb") as f:
            cpk.dump(inference, f)
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
    main(hparams, hparams_cnn, training_parameters, training_data)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
