import numpy as np
import torch


def load_data_wide(hparams):
    """Load the wide-grid training subset used for NPE training.

    Applies the fixed train split from ``training_set_1000_index.npy`` and
    returns tensors on ``hparams.device``.
    """

    training_data = np.load('/pscratch/sd/l/lgong/C3NN_inference/old_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    #training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    training_index = np.load('training_set_1000_index.npy') # perform a 1000/250 split for the training and SBC test
    training_data = training_data[training_index]
    #training_data = torch.tensor(training_data, dtype=torch.float32).to(device=hparams.device)
    training_data = torch.tensor(training_data, dtype=torch.float32)
    
    #param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = training_parameters[training_index]
    #training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device=hparams.device)
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)
    
    """
    # For reading in different perturbed data sets
    training_data1 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_data2 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization1.npy')
    #training_data3 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization2.npy')
    training_data = np.concatenate([training_data1, training_data2], axis=0)
    training_data = torch.tensor(training_data, dtype=torch.float32).to(device=hparams.device)

    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    training_parameters = np.array([np.load('CosmoGrid_params_wide_perturbed.npz')[par] for par in param_names]).T
    training_parameters = np.concatenate([training_parameters[0,:,:], training_parameters[1,:,:]], axis=0)
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32).to(device=hparams.device)
    """
    
    return training_parameters, training_data

def load_data_narrow(hparams):
    """Load the narrow-grid cosmologies used for extended training sets."""

    training_data = np.load('/pscratch/sd/l/lgong/C3NN_inference/old_data/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')#[50:]
    #training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')#[50:]
    training_data = torch.tensor(training_data, dtype=torch.float32)

    param_names = ['Om', 'sigma8', 'w0']
    #training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][1300:] for par in param_names]).T
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][1250:] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)

    return training_parameters, training_data

def load_data_wide_realization1(hparams):
    """Load wide-grid data from realization 1 without sub-selection."""

    training_data = np.load('/pscratch/sd/l/lgong/C3NN_inference/old_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization1.npy')
    #training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization1.npy')
    training_data = training_data#[training_index]
    training_data = torch.tensor(training_data, dtype=torch.float32)

    #param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = training_parameters#[training_index]
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)
    
    
    return training_parameters, training_data

def load_data_narrow_realization1(hparams):
    """Load narrow-grid data from realization 1."""

    training_data = np.load('/pscratch/sd/l/lgong/C3NN_inference/old_data/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')#[50:]
    #training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization1.npy')#[50:]
    training_data = torch.tensor(training_data, dtype=torch.float32)

    param_names = ['Om', 'sigma8', 'w0']
    #training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][1300:] for par in param_names]).T
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][1250:] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)

    return training_parameters, training_data

def load_data_wide_realization2(hparams):
    """Load wide-grid data from realization 2 without sub-selection."""

    #training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization2.npy')
    training_data = training_data#[training_index]
    training_data = torch.tensor(training_data, dtype=torch.float32)

    #param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = training_parameters#[training_index]
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)
    
    
    return training_parameters, training_data

def load_data_narrow_realization2(hparams):
    """Load narrow-grid data from realization 2."""

    #training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')[50:]
    training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization2.npy')#[50:]
    training_data = torch.tensor(training_data, dtype=torch.float32)

    param_names = ['Om', 'sigma8', 'w0']
    #training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][1300:] for par in param_names]).T
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][1250:] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)

    return training_parameters, training_data
def load_data_full(hparams):
    """Load concatenated wide+narrow realization-0 data from legacy paths."""

    training_data1 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_data2 = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_narrow_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_data = np.concatenate([training_data1, training_data2], axis=0)   
    training_data = torch.tensor(training_data, dtype=torch.float32)

    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)

    return training_parameters, training_data

def load_data_wide_noaverage(hparams):
    """Load wide-grid maps flattened over realizations (legacy helper)."""

    training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_index = np.load('training_set_1000_index.npy') # perform a 1000/250 split for the training and SBC test
    training_data = training_data[training_index]
    training_data = training_data.reshape(-1, 4, 50, 50)
    training_data = torch.tensor(training_data, dtype=torch.float32)

    #param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = training_parameters[training_index]
    training_parameters = np.repeat(training_parameters, 150, axis=0)
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)

    return training_parameters, training_data

'''
def load_data_hybrid(hparams):
    """Load hybrid inputs by appending compressed 2PCF features as channel 5."""

    training_data = np.load('/pscratch/sd/l/lgong/C3NN_inference/old_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    #training_data = np.load('/project/ag-weller/Kai.Lehman/C3NN_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection.npy')
    training_data = torch.tensor(training_data, dtype=torch.float32)[:1250]
    #training_data = torch.tensor(training_data, dtype=torch.float32)

    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters[:,:3], dtype=torch.float32)
    
    num_cosmo, num_maps, num_channels, num_pixels_side, _ = training_data.shape

    #reading in the learned 2pt compression
    compressed_2pt = np.load('/pscratch/sd/l/lgong/C3NN_inference/old_data/compressed_2pt.npy')
    compressed_2pt_reshaped = torch.tensor(np.repeat(compressed_2pt,num_maps,axis=0).reshape((num_cosmo,num_maps,8)))

    #reshaping the two point such that it is another channel for each map. Padded with zeros
    padded = torch.cat((compressed_2pt_reshaped,torch.tensor(0).repeat((num_cosmo,num_maps,num_pixels_side*num_pixels_side-8))),axis=2)
    padded_square = padded.reshape((num_cosmo,num_maps,num_pixels_side,num_pixels_side)).unsqueeze(2)

    #add 2pt as another channel 
    maps_and_2pt = torch.cat((training_data,padded_square),axis=2)
    
    return training_parameters, maps_and_2pt
'''

def load_data_hybrid(hparams):
    """Load hybrid inputs by appending compressed 2PCF features as channel 5."""

    # 1. Load and cast map data
    training_data = np.load('/pscratch/sd/l/lgong/C3NN_inference/old_data/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_data = torch.tensor(training_data[:1250], dtype=torch.float32)

    # 2. Load and cast parameters
    param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    # NOTE: Make sure 'CosmoGrid_params.npz' is in your current working directory when running!
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = torch.tensor(training_parameters[:,:3], dtype=torch.float32)
    
    num_cosmo, num_maps, num_channels, num_pixels_side, _ = training_data.shape

    # 3. Load 2pt compression and cast to float32 IMMEDIATELY
    compressed_2pt = np.load('/pscratch/sd/l/lgong/C3NN_inference/old_data/compressed_2pt.npy')
    compressed_2pt_reshaped = torch.tensor(
        np.repeat(compressed_2pt, num_maps, axis=0).reshape((num_cosmo, num_maps, 8)),
        dtype=torch.float32
    )

    # 4. Create zeros explicitly as float32 for padding
    zeros_pad = torch.zeros(
        (num_cosmo, num_maps, num_pixels_side * num_pixels_side - 8), 
        dtype=torch.float32
    )
    
    # 5. Concatenate and reshape (using 'dim' instead of 'axis' for standard PyTorch syntax)
    padded = torch.cat((compressed_2pt_reshaped, zeros_pad), dim=2)
    padded_square = padded.reshape((num_cosmo, num_maps, num_pixels_side, num_pixels_side)).unsqueeze(2)

    # 6. Add 2pt as the 5th channel 
    maps_and_2pt = torch.cat((training_data, padded_square), dim=2)
    
    return training_parameters, maps_and_2pt

def load_partitioned_mmap(map_file_path, partition_indices):
    """Opens mmap and loads corresponding parameters/2pt without RAM bloat."""
    print(f"Opening mmap: {map_file_path}...")
    
    # 1. Load as memmap (DO NOT cast to torch.tensor here)
    maps_mmap = np.load(map_file_path, mmap_mode='r')
    
    # 2. Slice parameters
    all_params = np.load('/pscratch/sd/l/lgong/C3NN_inference/data/full_parameters_2500nodes_Om_sigma8_w0_Aia_Mc_nu.npy')
    partition_params = all_params[partition_indices, :5]
    
    # 3. Slice 2pt compression
    all_compressed_2pt = np.load('/pscratch/sd/l/lgong/C3NN_inference/data/moped_coeffs_2PCF_grid.npy')
    partition_2pt = all_compressed_2pt[partition_indices]
    
    return maps_mmap, partition_params, partition_2pt

def load_data_wide_LSST(hparams):
    """Load LSST-scale (87x87) wide-grid maps from legacy storage."""

    training_data = np.load('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_87_by_87pixels_10_by_10deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy')
    training_index = np.load('training_set_1000_index.npy') # perform a 1000/250 split for the training and SBC test
    training_data = training_data[training_index]
    training_data = torch.tensor(training_data, dtype=torch.float32)

    #param_names = ['Om', 'sigma8', 'w0', 'ns', 'Ob', 'H0', 'bary_Mc', 'bary_nu']
    param_names = ['Om', 'sigma8', 'w0']
    training_parameters = np.array([np.load('CosmoGrid_params.npz')[par][:1250] for par in param_names]).T
    training_parameters = training_parameters[training_index]
    training_parameters = torch.tensor(training_parameters, dtype=torch.float32)
    
    return training_parameters, training_data
