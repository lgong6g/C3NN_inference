import time
import numpy as np
import gstools as gs
from gstools.random import MasterRNG

ens_no = 20
pixel_x = 100
pixel_y = 100
x = np.arange(pixel_x)
y = np.arange(pixel_y)
sigma2 = np.linspace(1,51,100)
corr_length = np.linspace(5,23,10)


for i in range(sigma2.shape[0]):
    for j in range(corr_length.shape[0]):
        start = time.time()

        print("sigma2 = %s"%(sigma2[i]))
        print("lambda = %s"%(corr_length[j]))
        
        model = gs.Gaussian(dim=2, var=sigma2[i], len_scale=corr_length[j])
        srf = gs.SRF(model)
        srf.set_pos([x, y], "structured")

        seed = MasterRNG(20170519)
        for k in range(ens_no):
            srf(seed=seed(), store=f"better_field{k}")
    
        grf_data = np.zeros((ens_no, pixel_x, pixel_y))
        for p in range(ens_no):
            grf_data[p,:,:] = srf[p]
            
        end = time.time()
        print("Generation of %s GRF maps takes %s seconds"%(ens_no, end-start))
        np.save('/e/ocean1/users/lgong/C3NN_inference/gstools_gaussian_random_fields/grf_%s_maps_%s_by_%s_pixels_var_%s_corrlen_%s.npy'%(ens_no, pixel_x, pixel_y, sigma2[i], corr_length[j]), grf_data)