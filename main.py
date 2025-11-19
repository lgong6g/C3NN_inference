import sys
sys.path.append('./src')
from hyperparameters import hparams
from C3NN_extensions import C3NN_w_std, C3NN_Nptonly_Nptbn, CombinedC3NN_Nptbn, C3NN_hybrid
from C3NN import C3NN
from load_data import load_data_wide, load_data_hybrid
from sbi.inference import SNPE_C
from sbi import utils
from sbi.utils.get_nn_models import posterior_nn
import numpy as np
import random
from torch import from_numpy
import torch
import time, datetime
import cloudpickle as cpk
import argparse



def main(hparams, training_parameters, training_data, verbose=True, test = False):

    # Make runs repeatable
    random_seed = 1750
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = True  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # instatiate model, optimizer and loss function
    print('instantiating model...')
    if hparams.model == 'default':
        c3nn = C3NN(hparams).to(device='cuda')
    elif hparams.model == 'w_std':
        c3nn = C3NN_w_std(hparams).to(device='cuda')
    elif hparams.model == 'hybrid':
        c3nn = C3NN_hybrid(hparams).to(device='cuda')
    else:
        raise NotImplementedError(f"Name '{hparams.model}' not implemented.")


    #c3nn = C3NN(in_channels=in_channels, num_filts=num_filts, filter_size=filter_size, order=order, absbeta=False).to(device='cuda')

    # Set up prior for sbi
    # The param here are Om, sigma8, w0, ns, Ob, H0, bary_Mc, bary_nu
    #lower = np.array([0.1, 0.4, -2.0, 0.87, 0.03, 64.0, 12.0, -2.0])
    #upper = np.array([0.5, 1.4, -0.33, 1.07, 0.06, 82.0, 15.0, 2.0])
    # now three dim prior
    lower = np.array([0.1, 0.4, -2.0])
    upper = np.array([0.5, 1.4, -0.33])
    prior = utils.BoxUniform(low=from_numpy(lower), high=from_numpy(upper),device=hparams.device)

    #set up sbi architecture
    density_estimator_custom = posterior_nn(model="maf", hidden_features=hparams.flow_hidden_features, num_transforms=hparams.flow_num_transforms, num_blocks=hparams.num_blocks, device=hparams.device, embedding_net=c3nn, z_score_x=None)
    inference = SNPE_C(prior, device=hparams.device, density_estimator=density_estimator_custom)

    # joint training
    if test:
        density_estimator = inference.append_simulations(training_parameters, training_data).train(learning_rate=hparams.learning_rate, training_batch_size=hparams.flow_training_batch_size, max_num_epochs=3)
    else:
        density_estimator = inference.append_simulations(training_parameters, training_data).train(learning_rate=hparams.learning_rate, training_batch_size=hparams.flow_training_batch_size, max_num_epochs=300)

    val_loss = inference.summary['best_validation_log_prob']

    if hparams.save:
        #model_dir = f"/e/ocean1/users/lgong/C3NN_inference/c3nn-sbi_models/c3nn_tuned_model_param_3D_25filter_pix37_2order_nozscorex_bn_w_std_no_c1/trial_{trial.number}"
        #model_dir = f"/home/k/Kai.Lehman/C3NN_inference/Models/c3nn_tuned_model_param_8D_40filter_3order_nozscorex_bn/trial_{trial.number}"
        if test:
            model_dir = f"/home/k/Kai.Lehman/C3NN_inference/Models/codetesting"
        else:
            model_dir = f"/home/k/Kai.Lehman/C3NN_inference/Models"
        #os.makedirs(model_dir, exist_ok=True)
        with open(f"{model_dir}/{hparams.name_model()}_inference.pkl", "wb") as handle:
            cpk.dump(inference, handle)
        with open(f"{model_dir}/{hparams.name_model()}_density_estimator.pkl", "wb") as handle:
            cpk.dump(density_estimator, handle)
        print('Model saved')
    else:
        print('Not saving')

    if verbose:
        print(val_loss)

    return val_loss, density_estimator, inference




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="My program description")
    parser.add_argument("-t", "--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    if args.test:
        print('Running in test mode')

    time_ini = time.time()

    if hparams.model == 'hybrid':
        training_parameters, training_data = load_data_hybrid(hparams)
    else:
        training_parameters, training_data = load_data_wide(hparams)
        #training_parameters, training_data = load_data_narrow(hparams)

    if args.test:
        training_parameters, training_data = training_parameters[:10], training_data[:10]

    main(hparams, training_parameters, training_data, test= args.test)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
