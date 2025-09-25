import numpy as np
import matplotlib.pyplot as plt
from treecorr import kkcorrelation, Catalog
from itertools import combinations_with_replacement
from tqdm import tqdm
import psutil
import os
import gc
import subprocess
import sys
import tempfile


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


def create_cosmology_script(cosmo, data_path, output_file):
    """Create a standalone script to process a single cosmology"""
    script_content = f"""
import numpy as np
from treecorr import kkcorrelation, Catalog
from itertools import combinations_with_replacement
import os
import gc

def process_cosmology():
    # Load data
    square_maps = np.load('{data_path}', mmap_mode='r')
    
    # Set up coordinates
    x = np.arange(50) * (6 * 60 / 50)  # arcmin
    y = np.arange(50) * (6 * 60 / 50)  # arcmin
    X, Y = np.meshgrid(x, y)
    x_coords = X.flatten()
    y_coords = Y.flatten()
    
    # Get redshift combinations
    zcombs = list(combinations_with_replacement(range(4), 2))
    
    cosmo = {cosmo}
    cosmo_corrs = np.zeros((150, 10, 10))
    
    for map_id in range(150):
        for i, (z1, z2) in enumerate(zcombs):
            # Create copies of the data
            k1_data = square_maps[cosmo, map_id, z1].flatten().copy()
            k2_data = square_maps[cosmo, map_id, z2].flatten().copy()
            
            # Create fresh correlation object
            corr = kkcorrelation.KKCorrelation(
                min_sep=10, 
                max_sep=250, 
                nbins=10, 
                sep_units='arcmin', 
                bin_slop=0.1, 
                metric='Euclidean'
            )
            
            cat1 = Catalog(
                x=x_coords, 
                y=y_coords, 
                k=k1_data, 
                x_units='arcmin', 
                y_units='arcmin'
            )
            cat2 = Catalog(
                x=x_coords, 
                y=y_coords, 
                k=k2_data, 
                x_units='arcmin', 
                y_units='arcmin'
            )
            
            corr.process(cat1, cat2)
            xi_value = corr.xi.copy()
            
            # Clean up
            del cat1, cat2, k1_data, k2_data, corr
            gc.collect()
            
            cosmo_corrs[map_id, i] = xi_value
    
    # Save result
    np.save('{output_file}', cosmo_corrs)
    print(f"Completed cosmology {{cosmo}}")

if __name__ == "__main__":
    process_cosmology()
"""
    return script_content


def process_cosmology_subprocess(cosmo, data_path, output_file):
    """Process a single cosmology in a completely separate Python process"""
    # Create temporary script file
    script_content = create_cosmology_script(cosmo, data_path, output_file)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the script in a subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"Error processing cosmology {cosmo}: {result.stderr}")
            return False
        
        # Check if output file was created
        if os.path.exists(output_file):
            return True
        else:
            print(f"Output file not created for cosmology {cosmo}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout processing cosmology {cosmo}")
        return False
    except Exception as e:
        print(f"Exception processing cosmology {cosmo}: {e}")
        return False
    finally:
        # Clean up script file
        if os.path.exists(script_path):
            os.remove(script_path)
    
    return False


def compute_correlations_subprocess(data_path):
    """Compute correlations using subprocess isolation"""
    print("Starting subprocess-based correlation computation...")
    print(f"Memory usage: {get_memory_usage():.2f} GB")
    
    # Create output directory
    os.makedirs('./data', exist_ok=True)
    
    start_id = 0
    stop_id = 1250
    
    successful_cosmologies = 0
    failed_cosmologies = 0
    
    # Create progress bar
    with tqdm(total=stop_id - start_id, desc="Processing cosmologies") as pbar:
        for cosmo in range(start_id, stop_id):
            output_file = f'./data/2PCF_wide_realization0_cosmo{cosmo}.npy'
            
            pbar.set_description(f'Processing cosmology {cosmo}')
            
            success = process_cosmology_subprocess(cosmo, data_path, output_file)
            
            if success:
                successful_cosmologies += 1
            else:
                failed_cosmologies += 1
            
            # Update progress bar with additional info
            pbar.set_postfix(
                Memory=f"{get_memory_usage():.2f} GB",
                Success=successful_cosmologies,
                Failed=failed_cosmologies
            )
            pbar.update(1)
    
    print("Subprocess correlation computation completed!")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    print(f"Total successful: {successful_cosmologies}, Total failed: {failed_cosmologies}")


def compute_mean_correlations():
    """Load individual cosmology files and compute mean correlations"""
    print("Computing mean correlations...")
    print(f"Memory usage: {get_memory_usage():.2f} GB")
    
    corrs = np.zeros((1250, 150, 10, 10))
    
    with tqdm(total=1250, desc="Loading cosmologies") as pbar:
        for cosmo in range(1250):
            file_path = f'./data/2PCF_wide_realization0_cosmo{cosmo}.npy'
            if os.path.exists(file_path):
                corrs[cosmo] = np.load(file_path)
                # Clean up individual files to save disk space
                os.remove(file_path)
            else:
                print(f"Warning: File not found for cosmology {cosmo}")
            
            pbar.set_description(f'Loading cosmology {cosmo}')
            pbar.set_postfix(Memory=f"{get_memory_usage():.2f} GB")
            pbar.update(1)
    
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
    print("Starting 2PCF calculation with complete subprocess isolation...")
    
    # Data path
    data_path = '/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization0.npy'
    
    try:
        # Compute correlations using subprocess isolation
        compute_correlations_subprocess(data_path)
        
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
