import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import skimage.io as io
from skimage import color
import os
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform
import PLIEqs as plieq



#root_dir = '/home/maryana/storage/Posdoc/PLI/2017/10181.10_Optic_Chiasm/PLI/raw/TIFF'
root_dir = '/home/maryana/storage/Posdoc/PLI/2017/2372.10_Optic_Chiasm/PLI/raw/TIFF'
y = range(18)
rrate = 0.25

img_name = 'slice_{:04d}_{:02d}.tif'
mask_name = 'mask_nerve1_slice_{:04d}_{:02d}.tif'
nAng = 18
nSlice = 10

tmp = io.imread(os.path.join(root_dir,img_name.format(nSlice,0)))
#tmp = xform.rescale(tmp,rrate,preserve_range=True)
mask = io.imread(os.path.join(root_dir,mask_name.format(nSlice,0)))
#mask = xform.rescale(mask,rrate,preserve_range=True)
if mask.ndim > 2:
    mask = mask[...,0]
idx_r,idx_c = np.nonzero(mask > 0)
#nPix = len(idx_r)
vol = np.zeros([tmp.shape[0],tmp.shape[1],nAng])

#load data
for ang in range(nAng):
    img = io.imread(os.path.join(root_dir,img_name.format(nSlice,ang)))
    #img = xform.rescale(img,rrate,preserve_range=True)
    vol[...,ang] = img[...,1]

#do computations
a0 = plieq.compute_a0(vol)
I0 = plieq.compute_I0(a0)
a2 = plieq.compute_a2(vol)
b2 = plieq.compute_b2(vol)
dir_map2 = plieq.compute_direction_eq(a2,b2)
sind_map2 = plieq.compute_sind_eq(a0,a2,b2)

sind_map = np.zeros(I0.shape)
dir_map = np.zeros(I0.shape)
#inc_map = np.zeros(I0.shape)

ang = np.arange(0, 180, 10)
t = np.radians(ang)
angles = np.arange(0,180)
t2 = np.radians(angles)
#idx_rows,idx_cols = np.nonzero(mask > 0)
# idx_rows = np.arange(0,vol.shape[0])
# idx_cols = np.arange(0,vol.shape[1])

nPix = len(idx_r)
for n in range(nPix):
    r = idx_r[n]
    c = idx_c[n]
    data = vol[r,c,:]
    try:
        fit,data_guess = plieq.fit_curve(data,t)
        data_fit = plieq.sin_curve(t2, *fit[0])
        i0 = I0[r,c]
        sd = (data_fit.max() - data_fit.min()) / i0
        #|sin delta|
        sind_map[r,c] = sd
        #direction map theta
        idx_min = np.argmin(data_fit, axis=0)
        a = angles[idx_min]
        dir_map[r,c] = a
    except RuntimeError:
        print('Error')
        plt.plot(data)

l = 565 #565nm
d = 70000 #190um in nm
RI = 1.47 #from Axers paper
delta = plieq.compute_delta(sind_map)
inc_map = plieq.compute_alpha(l,d,RI,delta)
inc_map = np.rad2deg(inc_map)

plt.imshow(inc_map)