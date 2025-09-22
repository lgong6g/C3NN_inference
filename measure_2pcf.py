import numpy as np
import matplotlib.pyplot as plt
from treecorr import kkcorrelation, Catalog
import torch
from itertools import combinations_with_replacement
from tqdm import tqdm
import psutil
import os
import gc


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


def load_data(data_path):
    """Load the kappa maps data using memory mapping"""
    print(f"Loading data from {data_path}")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    square_maps = np.load(data_path, mmap_mode='r')
    print(f"Data shape: {square_maps.shape}")  # (1250, 150, 4, 50, 50)
    print(f"Memory usage after loading: {get_memory_usage():.2f} GB")
    
    return square_maps


def setup_coordinates():
    """Set up coordinate arrays for the correlation calculation"""
    x = np.arange(50) * (6 * 60 / 50)  # arcmin
    y = np.arange(50) * (6 * 60 / 50)  # arcmin
    
    X, Y = np.meshgrid(x, y)
    x_coords = X.flatten()
    y_coords = Y.flatten()
    
    return x_coords, y_coords


def get_redshift_combinations():
    """Get all redshift bin combinations"""
    zcombs = list(combinations_with_replacement(range(4), 2))
    print(f"Redshift combinations: {zcombs}")
    return zcombs


def create_correlation_object():
    """Create and return a TreeCorr KKCorrelation object"""
    return kkcorrelation.KKCorrelation(
        min_sep=10, 
        max_sep=250, 
        nbins=20, 
        sep_units='arcmin', 
        bin_slop=0.1, 
        metric='Euclidean'
    )


def process_single_correlation(square_maps, cosmo, map_id, z1, z2, x_coords, y_coords, corr):
    """Process a single correlation calculation"""
    cat1 = Catalog(
        x=x_coords, 
        y=y_coords, 
        k=square_maps[cosmo, map_id, z1].flatten(), 
        x_units='arcmin', 
        y_units='arcmin'
    )
    cat2 = Catalog(
        x=x_coords, 
        y=y_coords, 
        k=square_maps[cosmo, map_id, z2].flatten(), 
        x_units='arcmin', 
        y_units='arcmin'
    )
    
    corr.process(cat1, cat2)
    xi_value = corr.xi
    corr.clear()
    
    # Clean up catalogs
    del cat1, cat2
    
    return xi_value


def process_cosmology(square_maps, cosmo, x_coords, y_coords, zcombs, corr, pbar):
    """Process all correlations for a single cosmology"""
    cosmo_corrs = np.zeros((150, 10, 20))
    
    for map_id in range(150):
        for i, (z1, z2) in enumerate(zcombs):
            xi_value = process_single_correlation(
                square_maps, cosmo, map_id, z1, z2, 
                x_coords, y_coords, corr
            )
            cosmo_corrs[map_id, i] = xi_value
            
            pbar.update(1)
            pbar.set_description(f'Processing cosmology {cosmo}, map {map_id}, zbin {i}')
            pbar.set_postfix(Memory=f"{get_memory_usage():.2f} GB")
    
    return cosmo_corrs


def compute_correlations(square_maps, x_coords, y_coords, zcombs):
    """Main function to compute all correlations"""
    print("Starting correlation computation...")
    print(f"Memory usage: {get_memory_usage():.2f} GB")
    
    # Create output directory
    os.makedirs('./data', exist_ok=True)
    
    # Create correlation object
    corr = create_correlation_object()
    
    total_operations = 1250 * 150 * len(zcombs)
    
    with tqdm(total=total_operations) as pbar:
        for cosmo in range(1250):
            # Recreate correlation object every 100 cosmologies to manage memory
            if cosmo % 100 == 0 and cosmo > 0:
                del corr
                gc.collect()
                corr = create_correlation_object()
                print(f"Recreated correlation object at cosmology {cosmo}")
                print(f"Memory usage: {get_memory_usage():.2f} GB")
            
            # Process this cosmology
            cosmo_corrs = process_cosmology(
                square_maps, cosmo, x_coords, y_coords, zcombs, corr, pbar
            )
            
            # Save results
            np.save(f'./data/2PCF_wide_realization0_cosmo{cosmo}.npy', cosmo_corrs)
            
            # Clean up
            del cosmo_corrs
            gc.collect()
    
    # Clean up correlation object
    del corr
    gc.collect()
    
    print("Correlation computation completed!")
    print(f"Memory usage: {get_memory_usage():.2f} GB")


def compute_mean_correlations():
    """Load individual cosmology files and compute mean correlations"""
    print("Computing mean correlations...")
    print(f"Memory usage: {get_memory_usage():.2f} GB")
    
    corrs = np.zeros((1250, 150, 10, 20))
    
    for cosmo in range(1250):
        corrs[cosmo] = np.load(f'./data/2PCF_wide_realization0_cosmo{cosmo}.npy')
        
        # Clean up individual files to save disk space
        os.remove(f'./data/2PCF_wide_realization0_cosmo{cosmo}.npy')
        
        if cosmo % 100 == 0:
            print(f"Loaded cosmology {cosmo}, Memory usage: {get_memory_usage():.2f} GB")
    
    # Compute mean across maps
    corrs_mean = np.mean(corrs, axis=1)
    np.save('./data/2PCF_wide_realization0_mean.npy', corrs_mean)
    
    print("Mean correlations saved!")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    # Clean up
    del corrs, corrs_mean
    gc.collect()


def main():
    """Main function to calculate 2PCFs"""
    print("Starting 2PCF calculation...")
    
    # Data path
    data_path = '/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy'
    
    try:
        # Load data
        square_maps = load_data(data_path)
        
        # Set up coordinates
        x_coords, y_coords = setup_coordinates()
        
        # Get redshift combinations
        zcombs = get_redshift_combinations()
        
        # Compute correlations
        compute_correlations(square_maps, x_coords, y_coords, zcombs)
        
        # Clean up data
        del square_maps, x_coords, y_coords, zcombs
        gc.collect()
        
        # Compute mean correlations
        compute_mean_correlations()
        
        print("2PCF calculation completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    finally:
        # Final cleanup
        gc.collect()
        print(f"Final memory usage: {get_memory_usage():.2f} GB")


if __name__ == "__main__":
    main()