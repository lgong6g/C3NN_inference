from functools import partial
from astropy.io import fits

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

############################## fibonacci sphere ################################
# Distribute n points on a sphere with the Fibonacci algorithm, returns the    #
# positions of such points.                                                    #
# More on the algorithm at http://extremelearning.com.au                       #
#                                                                              #
# Parameters:                                                                  #
# - n (int): number of points to place on the sphere                           #
# - lonlat (bool): wheter the output has to be as longitude and latitude in    #
#                  degrees or colatitude and longitude in radians.             #
#                                                                              #
# Returns:                                                                     #
# - coords (2D array): array with shape n x 2, each row representing a point   #
################################################################################


def fibonacci_sphere(n, lonlat=True):
    goldenRatio = (1 + 5**0.5) / 2
    i = np.arange(0, n)
    phi = (2 * np.pi * i / goldenRatio) % (2 * np.pi)
    theta = np.arccos(1 - 2 * (i + 0.5) / n)
    if lonlat:
        coords = np.stack((np.degrees(phi), np.degrees(theta - np.pi / 2)), axis=-1)
    else:
        coords = np.stack((theta, phi), axis=-1)
    return coords

# Set the size of the squares in degrees
alpha = 50*hp.nside2resol(512, True)/60
# Set the resolution of the HEALPix map
NSIDE = 512
# Set the number of squares you want to distribute on the sphere
nsquares = 152
# Set how many squares you want to skip (e.g. nskip = 1 will remove the first and the last square)
nskip = 1

 
NPIX = hp.nside2npix(NSIDE)
coords = fibonacci_sphere(nsquares, lonlat=True)
m = np.zeros(NPIX, dtype=int)
lon, lat = coords[nskip:-nskip, 0], coords[nskip:-nskip, 1]

coords_update = coords[nskip:-nskip, :]
nsquares_update = int(nsquares - 2*nskip)

# Set up number of pixels along one side of the square
n = 50
xsize=n
reso=(alpha * 60) / n


for i in range(len(lon)):
    # We get the vertices of each square (notice thet they are tilted by 45 deg)
    proj = hp.projector.GnomonicProj(rot=(lon[i], lat[i], 45), xsize=n, ysize=None, reso=reso)
    vecs = []
    for idx in [(0,0),(0,n-1),(n-1,n-1),(n-1,0)]:
        xy = proj.ij2xy(idx)
        vecs.append(proj.xy2vec(xy[0], xy[1]))
    vertices = np.array(vecs)
    # We use the vertices to get the pixel indices that are inside the square
    idx = hp.query_polygon(NSIDE, vertices, inclusive=True)
    m[idx] += 1
    

print(f'number of maps {len(lon)}')
print(f'covered area {round((len(lon)*alpha**2)/41253, 3)*100}%')
print(f'overlapping pixels: {np.sum(m==2)}')

################################ extract region ################################
# Returns a n x n matrix containing the values of the map m inside a square    #
# patch. The square patch is centered at (lon, lat) and has a side of alpha    #
# degrees.                                                                     #
#                                                                              #
# Parameters:                                                                  #
# - m (array-like): healpix map from which extract the region                  #
# - lon (array-like or int): longitudes of the maps centers, in degrees        #
# - lat (array-like or int): latitudes of the maps centers, in degrees         #
# - phi (array-like or int): rotation angle, in degrees                        #
# - alpha (float): angular size of the maps, in degrees                        #
# - n (int): number of pixels per side of the region                           #
# - fout (str or None): path of the folder which to save the region            #
# - nest (bool): whether the angular maps are nest ordered or not              #
#                                                                              #
# Returns:                                                                     #
# - region (2D array): 2D array containing the region extracted                #
################################################################################


def extract_region(m, lon, lat, phi, alpha, n, fout=None, nest=False):
    # Compute region
    nside = hp.npix2nside(len(m))
    proj = hp.projector.GnomonicProj(
        rot=(lon, lat, phi), xsize=n, reso=(alpha * 60) / n
    )
    #region = proj.projmap(m, vec2pix_func=partial(hp.vec2pix, nside))[:][::-1]
    region = proj.projmap(m, vec2pix_func=partial(hp.vec2pix, nside))
    # Eventually save region in numpy format
    if fout:
        print(f"Saved region to {fout}")
        np.save(fout, region)
    return region

sobol_index = np.loadtxt('../sobol_indices_cosmogridv1.txt')
wide_index = sobol_index[:1250]
#narrow_index = sobol_index[1250:]
n_maps = wide_index.shape[0]
n_pixels = n
#smooth_scale = 10 # in unit of arcmin
gnom_proj_array = np.zeros((n_maps, nsquares_update, 4, n_pixels, n_pixels))
k = 0
for m in wide_index:
    # Maps with only weak lensing convergence signal
    # For the test data set we use the realization index 0
    kappa_map = np.load("/e/ocean1/users/dgebauer/c3nn/kappa_maps/grid_baryonified/cosmo_%06d/nzmap_2.npy"%(m))
    kappa_map_bin1 = kappa_map[0]
    kappa_map_bin2 = kappa_map[1]
    kappa_map_bin3 = kappa_map[2]
    kappa_map_bin4 = kappa_map[3]

    # Maps with Gaussian shape noise
    #noise_map = np.load('/data/taurus/users/dgebauer/CCNN_maps/cosmo_delta_w0_p/shape_noise/noise_0%03d.npy'%(m))
    #noise_map_bin1 = noise_map[0]
    #noise_map_bin2 = noise_map[1]
    #noise_map_bin3 = noise_map[2]
    #noise_map_bin4 = noise_map[3]

    #kappa_map_noisy_bin1 = (kappa_map_bin1 + noise_map_bin1).astype(np.float32)
    #kappa_map_noisy_bin2 = (kappa_map_bin2 + noise_map_bin2).astype(np.float32)
    #kappa_map_noisy_bin3 = (kappa_map_bin3 + noise_map_bin3).astype(np.float32)
    #kappa_map_noisy_bin4 = (kappa_map_bin4 + noise_map_bin4).astype(np.float32)

    # We want to smooth the CosmogridV1 maps for C3NN with sigma = 7, 14 and 28 arcmin
    #kappa_map_noisy_bin1 = hp.sphtfunc.smoothing(kappa_map_noisy_bin1, fwhm=smooth_scale/(60*180)*np.pi)
    #kappa_map_noisy_bin2 = hp.sphtfunc.smoothing(kappa_map_noisy_bin2, fwhm=smooth_scale/(60*180)*np.pi)
    #kappa_map_noisy_bin3 = hp.sphtfunc.smoothing(kappa_map_noisy_bin3, fwhm=smooth_scale/(60*180)*np.pi)
    #kappa_map_noisy_bin4 = hp.sphtfunc.smoothing(kappa_map_noisy_bin4, fwhm=smooth_scale/(60*180)*np.pi)
    
    for i in range(coords_update.shape[0]):
        region_bin1 = extract_region(kappa_map_bin1, coords_update[i,0], coords_update[i,1], phi=45, alpha=50*hp.nside2resol(512, True)/60, n=n_pixels)
        region_bin2 = extract_region(kappa_map_bin2, coords_update[i,0], coords_update[i,1], phi=45, alpha=50*hp.nside2resol(512, True)/60, n=n_pixels)
        region_bin3 = extract_region(kappa_map_bin3, coords_update[i,0], coords_update[i,1], phi=45, alpha=50*hp.nside2resol(512, True)/60, n=n_pixels)
        region_bin4 = extract_region(kappa_map_bin4, coords_update[i,0], coords_update[i,1], phi=45, alpha=50*hp.nside2resol(512, True)/60, n=n_pixels)
        gnom_proj_array[k,i,0,:,:] = region_bin1
        gnom_proj_array[k,i,1,:,:] = region_bin2
        gnom_proj_array[k,i,2,:,:] = region_bin3
        gnom_proj_array[k,i,3,:,:] = region_bin4
        
    k += 1
    print("calculation complete on cosmology node %s"%(m))
    
#np.save('/e/ocean1/users/lgong/CCNN/takahashi/kappa_data_%s_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy'%(n_maps*len(lon)), gnom_proj_array)
np.save('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noiseless_1250_wide_grid_cosmologies_150_maps_50_by_50pixels_6_by_6deg_no_smooth_4_des_zsbin_gnom_projection_realization2.npy', gnom_proj_array)