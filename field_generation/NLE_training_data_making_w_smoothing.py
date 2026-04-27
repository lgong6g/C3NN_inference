import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import healpy as hp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

data_dir = "/pscratch/sd/d/dgebauer/C3NN/CosmoGrid/grid_baryonified"
mask = np.load('/pscratch/sd/d/dgebauer/C3NN/CosmoGrid/parameters/DESY3_Mask_512.npy')
n_files = 2500
n_realizations_per_file = 28
n_channels = 4
n_patches = 3
nside = 512
npix_per_base = nside**2

top_3_base_pixels = [8, 4, 11] # three NSIDE=1 pixels that contain the most number of DES Y3 NSIDE=512 pixels

target_dtype = np.float32

idx = np.arange(npix_per_base, dtype=np.int32)
x = np.zeros_like(idx)
y = np.zeros_like(idx)

for bit in range(int(np.log2(nside))):
    x |= ((idx >> (2 * bit)) & 1) << bit
    y |= ((idx >> (2 * bit + 1)) & 1) << bit

ring_idx = hp.nest2ring(nside, np.arange(hp.nside2npix(nside)))

bool_mask = np.zeros(hp.nside2npix(nside), dtype=bool)
bool_mask[mask] = True

full_mask = np.zeros(hp.nside2npix(nside), dtype=target_dtype)
full_mask[mask] = 1.0

sigma_arcmin = [43.70, 24.89, 16.27, 14.35]
sigma_rad = [f * np.pi / (180.0 * 60.0) for f in sigma_arcmin]

print("Pre-computing smoothed masks for normalized smoothing...")
smoothed_masks = []
for c in range(n_channels):
    # Smooth the binary mask
    s_mask = hp.smoothing(full_mask, sigma=sigma_rad[c])
    # Prevent divide-by-zero errors in empty space
    s_mask[s_mask < 1e-12] = 1e-12 
    smoothed_masks.append(s_mask.astype(target_dtype))

total_realizations = n_files * n_realizations_per_file
final_shape = (total_realizations, n_patches, n_channels, nside, nside)
output_filename = "/pscratch/sd/l/lgong/C3NN_inference/data/full_extracted_data_2500nodes_3patches_4zbins_512by512mmap_smoothed.npy"

if not os.path.exists(output_filename):
    print(f"Allocating {final_shape} array on disk as {target_dtype}...")
    np.lib.format.open_memmap(output_filename, mode='w+', dtype=target_dtype, shape=final_shape)

def process_file(file_idx):
    # Open memmap using the target_dtype
    local_mmap = np.lib.format.open_memmap(output_filename, mode='r+', dtype=target_dtype, shape=final_shape)
    
    filename = f"kappa_E_{file_idx:04d}.npy"
    filepath = os.path.join(data_dir, filename)
    
    # Load the array
    valid_kappa = np.load(filepath)

    kappa_full = np.zeros((7,4,4,hp.nside2npix(nside)), dtype=target_dtype)
    kappa_full[:,:,:,mask] = valid_kappa
    kappa = kappa_full.reshape((n_realizations_per_file, n_channels, hp.nside2npix(nside)))
    
    for r in range(n_realizations_per_file):
        for c in range(n_channels):
            # Smooth the data map
            s_data = hp.smoothing(kappa[r, c, :], sigma=sigma_rad[c])
            
            # Normalize by dividing by the smoothed mask
            smoothed_kappa = (s_data / smoothed_masks[c]).astype(target_dtype)
            
            # Re-zero all pixels safely outside the DES footprint
            smoothed_kappa[~bool_mask] = 0.0
            
            kappa[r, c, :] = smoothed_kappa
    
    kappa_nest = kappa[:, :, ring_idx] 

    r_start = file_idx * n_realizations_per_file
    
    for p, base_pix in enumerate(top_3_base_pixels):
        start_idx = base_pix * npix_per_base
        end_idx = (base_pix + 1) * npix_per_base
        
        chunk_1d = kappa_nest[:, :, start_idx:end_idx]
        
        for r in range(n_realizations_per_file):
            for c in range(n_channels):
                local_mmap[r_start + r, p, c, x, y] = chunk_1d[r, c, :]
                
    # Flush changes to disk
    local_mmap.flush()
    return file_idx

# --- 5. Execute in Parallel ---
if __name__ == '__main__':
    
    num_workers = 16 
    
    print(f"Starting parallel extraction with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, i) for i in range(n_files)]
        
        for _ in tqdm(as_completed(futures), total=n_files, desc="Processing Maps"):
            pass
            
    print(f"Extraction complete! Saved to {output_filename}")