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
alpha = 20
# Set the resolution of the HEALPix map
NSIDE = 2048
# Set the number of squares you want to distribute on the sphere
nsquares = 60
# Set how many squares you want to skip (e.g. nskip = 1 will remove the first and the last square)
nskip = 1

 
NPIX = hp.nside2npix(NSIDE)
coords = fibonacci_sphere(nsquares, lonlat=True)
m = np.zeros(NPIX, dtype=int)
lon, lat = coords[nskip:-nskip, 0], coords[nskip:-nskip, 1]

coords_update = coords[nskip:-nskip, :]
nsquares_update = int(nsquares - 2*nskip)

# Set up number of pixels along one side of the square
n = 200
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

n_maps = 200
n_pixels = n
smooth_scale = 10 # in unit of arcmin
gnom_proj_array = np.zeros((n_maps*nsquares_update, 4, n_pixels, n_pixels))
k = 0
for m in range(n_maps):
    #test_map_bin1 = hp.read_map('/data/taurus/users/WL_cosmology/data/simulations/Takahashi/realization%03d/weak_lensing_maps/Dirac_source_planes/allskymap_nres11r%03d.zs10.mag_kappa.fits'%(m,m))
    #test_map_bin2 = hp.read_map('/data/taurus/users/WL_cosmology/data/simulations/Takahashi/realization%03d/weak_lensing_maps/Dirac_source_planes/allskymap_nres11r%03d.zs16.mag_kappa.fits'%(m,m))
    
    #test_map_bin1 = hp.read_map('/e/ocean1/users/ahalder/data/simulations/flask_Takahashi_noiseless_CCNN_new_lambda/realization0%03d/gaussian_allskymap_nres11r0%03d.zs10.mag_kappa.fits'%(m,m))
    #test_map_bin2 = hp.read_map('/e/ocean1/users/ahalder/data/simulations/flask_Takahashi_noiseless_CCNN_new_lambda/realization0%03d/gaussian_allskymap_nres11r0%03d.zs16.mag_kappa.fits'%(m,m))

    
    # Maps with only weak lensing convergence signal
    kappa_map = np.load('/data/taurus/users/dgebauer/CCNN_maps/cosmo_delta_w0_p/kappa/nzmap_0%03d.npy'%(m))
    kappa_map_bin1 = kappa_map[0]
    kappa_map_bin2 = kappa_map[1]
    kappa_map_bin3 = kappa_map[2]
    kappa_map_bin4 = kappa_map[3]

    # Maps with Gaussian shape noise
    noise_map = np.load('/data/taurus/users/dgebauer/CCNN_maps/cosmo_delta_w0_p/shape_noise/noise_0%03d.npy'%(m))
    noise_map_bin1 = noise_map[0]
    noise_map_bin2 = noise_map[1]
    noise_map_bin3 = noise_map[2]
    noise_map_bin4 = noise_map[3]

    kappa_map_noisy_bin1 = (kappa_map_bin1 + noise_map_bin1).astype(np.float32)
    kappa_map_noisy_bin2 = (kappa_map_bin2 + noise_map_bin2).astype(np.float32)
    kappa_map_noisy_bin3 = (kappa_map_bin3 + noise_map_bin3).astype(np.float32)
    kappa_map_noisy_bin4 = (kappa_map_bin4 + noise_map_bin4).astype(np.float32)

    """
    kappa_map_noisy_bin1 = hp.sphtfunc.smoothing(kappa_map_noisy_bin1, fwhm=smooth_scale/(60*180)*np.pi)
    kappa_map_noisy_bin2 = hp.sphtfunc.smoothing(kappa_map_noisy_bin2, fwhm=smooth_scale/(60*180)*np.pi)
    kappa_map_noisy_bin3 = hp.sphtfunc.smoothing(kappa_map_noisy_bin3, fwhm=smooth_scale/(60*180)*np.pi)
    kappa_map_noisy_bin4 = hp.sphtfunc.smoothing(kappa_map_noisy_bin4, fwhm=smooth_scale/(60*180)*np.pi)
    """
    
    """
    # Maps with only intrinsic alignment signal 
    IA_map = np.load('/data/taurus/users/dgebauer/CCNN_maps/IA_alpha_4/nzmap_0%03d_IA.npy'%(m))
    IA_amp = 0.79
    IA_map_bin1 = IA_amp*IA_map[0]
    IA_map_bin2 = IA_amp*IA_map[1]
    IA_map_bin3 = IA_amp*IA_map[2]
    IA_map_bin4 = IA_amp*IA_map[3]

    # Maps of weak lensing convergence together with intrinsic alignment (NLA)
    wl_map_bin1 = kappa_map_bin1 + IA_map_bin1
    wl_map_bin2 = kappa_map_bin2 + IA_map_bin2
    wl_map_bin3 = kappa_map_bin3 + IA_map_bin3
    wl_map_bin4 = kappa_map_bin4 + IA_map_bin4
    """
    
    for i in range(coords_update.shape[0]):
        #region = extract_region(test_map, coords_update[i,0], coords_update[i,1], phi=45, alpha=20, n=n_pixels)
        #gnom_proj_array[k,:,:,:] = region
        
        #region_bin1 = extract_region(test_map_bin1, coords_update[i,0], coords_update[i,1], phi=45, alpha=20, n=n_pixels)
        #region_bin2 = extract_region(test_map_bin2, coords_update[i,0], coords_update[i,1], phi=45, alpha=20, n=n_pixels)
        #gnom_proj_array[k,0,:,:] = region_bin1
        #gnom_proj_array[k,1,:,:] = region_bin2

        
        region_bin1 = extract_region(kappa_map_bin1, coords_update[i,0], coords_update[i,1], phi=45, alpha=20, n=n_pixels)
        region_bin2 = extract_region(kappa_map_bin2, coords_update[i,0], coords_update[i,1], phi=45, alpha=20, n=n_pixels)
        region_bin3 = extract_region(kappa_map_bin3, coords_update[i,0], coords_update[i,1], phi=45, alpha=20, n=n_pixels)
        region_bin4 = extract_region(kappa_map_bin4, coords_update[i,0], coords_update[i,1], phi=45, alpha=20, n=n_pixels)
        gnom_proj_array[k,0,:,:] = region_bin1
        gnom_proj_array[k,1,:,:] = region_bin2
        gnom_proj_array[k,2,:,:] = region_bin3
        gnom_proj_array[k,3,:,:] = region_bin4
        
        
        k += 1

#np.save('/e/ocean1/users/lgong/CCNN/takahashi/kappa_data_%s_maps_200_by_200pixels_20_by_20deg_zs10_zs16_gnom_projection.npy'%(n_maps*len(lon)), gnom_proj_array)
np.save('/e/ocean1/users/lgong/CCNN/cosmogrid/kappa_noisy_delta_w0_plus_data_11600_maps_200_by_200pixels_20_by_20deg_no_smooth_4_des_zsbin_gnom_projection.npy', gnom_proj_array)