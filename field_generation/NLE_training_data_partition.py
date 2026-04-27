import os
import numpy as np
from tqdm import tqdm

'''
# --- 1. Configuration ---
data_dir = "/pscratch/sd/l/lgong/C3NN_inference/data"
master_map_file = os.path.join(data_dir, "full_extracted_data_2500nodes_3patches_4zbins_512by512mmap.npy")

# IMPORTANT: Update this with the actual path to your (70000, 6) parameter file
master_params_file = os.path.join(data_dir, "full_parameters_2500nodes_Om_sigma8_w0_Aia_Mc_nu.npy") 

total_realizations = 70000
nodes = 2500
realizations_per_node = 28
wide_nodes = 1300
narrow_nodes = 1200
target_dtype = np.float32

# --- 2. Determine Indices ---
print("Calculating split indices...")
np.random.seed(42) # Set seed for reproducible test set selection

# Randomly select 500 unique nodes from the first 1300 (wide grid)
test_nodes = set(np.random.choice(wide_nodes, 500, replace=False))

c3nn_indices = []
nle_indices = []
test_indices = []

for node in range(nodes):
    start_idx = node * realizations_per_node
    
    # C3NN embedding gets realizations 1-14 (local indices 0 to 13)
    c3nn_indices.extend(range(start_idx, start_idx + 14))
    
    if node in test_nodes:
        # NLE gets realizations 15-27
        nle_indices.extend(range(start_idx + 14, start_idx + 27))
        # Test gets the final realization (index 27) of this node
        test_indices.append(start_idx + 27)
    else:
        # NLE gets realizations 15-28
        nle_indices.extend(range(start_idx + 14, start_idx + 28))

# Convert to numpy arrays
c3nn_indices = np.array(c3nn_indices, dtype=np.int32)
nle_indices = np.array(nle_indices, dtype=np.int32)
test_indices = np.array(test_indices, dtype=np.int32)

print(f"C3NN Set Size: {len(c3nn_indices)}")
print(f"NLE Set Size: {len(nle_indices)}")
print(f"Test Set Size: {len(test_indices)}")

# Save the indices for future reference
np.save(os.path.join(data_dir, "c3nn_training_data_param_indices.npy"), c3nn_indices)
np.save(os.path.join(data_dir, "nle_training_data_param_indices.npy"), nle_indices)
np.save(os.path.join(data_dir, "test_data_param_indices.npy"), test_indices)

# --- 3. Split Parameter Array ---
print("Splitting and saving parameter arrays...")
master_params = np.load(master_params_file)

np.save(os.path.join(data_dir, "c3nn_training_params.npy"), master_params[c3nn_indices])
np.save(os.path.join(data_dir, "nle_training_params.npy"), master_params[nle_indices])
np.save(os.path.join(data_dir, "test_params.npy"), master_params[test_indices])

# --- 4. Allocate Disk Space for New Square Maps ---
print("Allocating memmap files on disk (this may take a moment)...")
c3nn_shape = (len(c3nn_indices), 3, 4, 512, 512)
nle_shape = (len(nle_indices), 3, 4, 512, 512)
test_shape = (len(test_indices), 3, 4, 512, 512)

c3nn_mmap = np.lib.format.open_memmap(os.path.join(data_dir, "c3nn_training_maps_mmap_unsmoothed.npy"), mode='w+', dtype=target_dtype, shape=c3nn_shape)
nle_mmap = np.lib.format.open_memmap(os.path.join(data_dir, "nle_training_maps_mmap_unsmoothed.npy"), mode='w+', dtype=target_dtype, shape=nle_shape)
test_mmap = np.lib.format.open_memmap(os.path.join(data_dir, "test_maps_mmap_unsmoothed.npy"), mode='w+', dtype=target_dtype, shape=test_shape)

# --- 5. Copy Data Sequentially (Optimized for Lustre I/O) ---
print("Opening master map file...")
master_mmap = np.load(master_map_file, mmap_mode='r')

# Convert lists to sets for O(1) lookup speed during the loop
c3nn_set = set(c3nn_indices)
nle_set = set(nle_indices)

# Pointers for writing
c_pt, n_pt, t_pt = 0, 0, 0

print("Copying data into split arrays...")
# We iterate sequentially through the 70,000 master array to prevent disk thrashing
for i in tqdm(range(total_realizations), desc="Partitioning 821GB Array"):
    if i in c3nn_set:
        c3nn_mmap[c_pt] = master_mmap[i]
        c_pt += 1
    elif i in nle_set:
        nle_mmap[n_pt] = master_mmap[i]
        n_pt += 1
    else:
        test_mmap[t_pt] = master_mmap[i]
        t_pt += 1

# Flush buffers to ensure all data is written
c3nn_mmap.flush()
nle_mmap.flush()
test_mmap.flush()

print("Data partitioning complete!")
'''

# Use the previously saved partition indices to slice the newly extracted smoothed square maps
data_dir = "/pscratch/sd/l/lgong/C3NN_inference/data"

# my newly extracted SMOOTHED 821 GB master file
master_map_file = os.path.join(data_dir, "full_extracted_data_2500nodes_3patches_4zbins_512by512mmap_smoothed.npy")

total_realizations = 70000
target_dtype = np.float32

print("Loading saved split indices...")
c3nn_indices = np.load(os.path.join(data_dir, "c3nn_training_data_param_indices.npy"))
nle_indices = np.load(os.path.join(data_dir, "nle_training_data_param_indices.npy"))
test_indices = np.load(os.path.join(data_dir, "test_data_param_indices.npy"))

print(f"Loaded C3NN Set Size: {len(c3nn_indices)}")
print(f"Loaded NLE Set Size: {len(nle_indices)}")
print(f"Loaded Test Set Size: {len(test_indices)}")

print("Allocating new memmap files on disk...")
c3nn_shape = (len(c3nn_indices), 3, 4, 512, 512)
nle_shape = (len(nle_indices), 3, 4, 512, 512)
test_shape = (len(test_indices), 3, 4, 512, 512)

c3nn_mmap = np.lib.format.open_memmap(os.path.join(data_dir, "c3nn_training_maps_mmap_smoothed.npy"), mode='w+', dtype=target_dtype, shape=c3nn_shape)
nle_mmap = np.lib.format.open_memmap(os.path.join(data_dir, "nle_training_maps_mmap_smoothed.npy"), mode='w+', dtype=target_dtype, shape=nle_shape)
test_mmap = np.lib.format.open_memmap(os.path.join(data_dir, "test_maps_mmap_smoothed.npy"), mode='w+', dtype=target_dtype, shape=test_shape)

print("Opening master smoothed map file...")
master_mmap = np.load(master_map_file, mmap_mode='r')

c3nn_set = set(c3nn_indices)
nle_set = set(nle_indices)

# Pointers to track the write position in each new array
c_pt, n_pt, t_pt = 0, 0, 0

print("Routing data into split arrays (Single Pass I/O)...")
for i in tqdm(range(total_realizations), desc="Partitioning Smoothed Maps"):
    if i in c3nn_set:
        c3nn_mmap[c_pt] = master_mmap[i]
        c_pt += 1
    elif i in nle_set:
        nle_mmap[n_pt] = master_mmap[i]
        n_pt += 1
    else:
        test_mmap[t_pt] = master_mmap[i]
        t_pt += 1

# Flush buffers to ensure Lustre writes everything safely to disk
c3nn_mmap.flush()
nle_mmap.flush()
test_mmap.flush()

print("Partitioning of smoothed data complete!")