import numpy as np
import torch


def load_data_wide(hparams):

    training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    #training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    training_index = np.load('training_set_1000_index.npy') # perform a 1000/250 split for the training and SBC test
    training_data = training_data[training_index]
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')

    training_parameters = np.load('CosmoGrid_params.npz').files
    #param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = training_parameters[training_index]
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')
    
    """
    # For reading in different perturbed data sets
    training_data1 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_data2 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization1.npy')
    #training_data3 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization2.npy')
    training_data = np.concatenate([training_data1, training_data2], axis=0)
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')

    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    training_parameters = np.array([np.load('CosmoGrid_params_wide_perturbed.npz')[par] for par in param_names]).T
    training_parameters = np.concatenate([training_parameters[0,:,:], training_parameters[1,:,:]], axis=0)
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')
    """
    
    return training_parameters, training_data

def load_data_narrow(hparams):

    #training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')[50:]
    training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')[50:]
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')

    training_parameters = np.load('CosmoGrid_params.npz').files
    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][1300:] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')

    return training_parameters, training_data

def load_data_full(hparams):

    training_data1 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_data2 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_data = np.concatenate([training_data1, training_data2], axis=0)   
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')

    training_parameters = np.load('CosmoGrid_params.npz').files
    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')

    return training_parameters, training_data

def load_data_wide_noaverage(hparams):

    training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_index = np.load('training_set_1000_index.npy') # perform a 1000/250 split for the training and SBC test
    training_data = training_data[training_index]
    training_data = training_data.reshape(-1, 4, 50, 50)
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')

    training_parameters = np.load('CosmoGrid_params.npz').files
    #param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = training_parameters[training_index]
    training_parameters = np.repeat(training_parameters, 150, axis=0)
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')

    return training_parameters, training_data


def load_data_hybrid(hparams):

    #training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    training_data = torch.tensor(training_data, dtype=torch.float32)[:1250].to(device=hparams.device)
    #training_data = torch.tensor(training_data, dtype=torch.float32)

    training_parameters = np.load('CosmoGrid_params.npz').files
    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters[:,:3], dtype=torch.float32).to(device=hparams.device)
    
    num_cosmo, num_maps, num_channels, num_pixels_side, _ = training_data.shape

    #reading in the learned 2pt compression
    compressed_2pt = np.load('./data/compressed_2pt.npy')
    compressed_2pt_reshaped = torch.tensor(np.repeat(compressed_2pt,num_maps,axis=0).reshape((num_cosmo,num_maps,8)))

    #reshaping the two point such that it is another channel for each map. Padded with zeros
    padded = torch.cat((compressed_2pt_reshaped,torch.tensor(0).repeat((num_cosmo,num_maps,num_pixels_side*num_pixels_side-8))),axis=2)
    padded_square = padded.reshape((num_cosmo,num_maps,num_pixels_side,num_pixels_side)).unsqueeze(2).to(device=hparams.device)

    #add 2pt as another channel 
    maps_and_2pt = torch.cat((training_data,padded_square),axis=2)
    
    return training_parameters, maps_and_2pt

def load_data_wide_LSST(hparams):

    training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_87_by_87pixels_10_by_10deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_index = np.load('training_set_1000_index.npy') # perform a 1000/250 split for the training and SBC test
    training_data = training_data[training_index]
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device='cuda')

    training_parameters = np.load('CosmoGrid_params.npz').files
    #param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = training_parameters[training_index]
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device='cuda')

    return training_parameters, training_data
