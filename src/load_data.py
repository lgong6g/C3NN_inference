import numpy as np
import torch

def load_data_wide(hparams):

    training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    #training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')
    #training_data = torch.tensor(training_data, dtype=torch.float32)

    training_parameters = np.load('CosmoGrid_params.npz').files
    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')
    #training_parameters = torch.tensor(training_parameters, dtype=torch.float32)

    return training_parameters, training_data

def load_data_narrow(hparams):

    training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')[50:]
    #training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')
    #training_data = torch.tensor(training_data, dtype=torch.float32)

    training_parameters = np.load('CosmoGrid_params.npz').files
    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][1300:] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')
    #training_parameters = torch.tensor(training_parameters, dtype=torch.float32)

    return training_parameters, training_data
