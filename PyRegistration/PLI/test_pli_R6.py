import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import rawpy
from scipy.optimize import curve_fit
import skimage.transform as xform



#root_dir = '/home/maryana/storage/Posdoc/PLI/2017/test_pli_R6'
root_dir='/media/maryana/SUSHI_HD/SUSHI/Posdoc/PLI/2017/test_pli_R6'
img_name = 'Image {:06d}.tif'
mask_name = 'mask_nerve4.tif'
nAngles = 18
nImgs = 7
vol = np.zeros([nAngles])

mask = io.imread(os.path.join(root_dir,mask_name.format(0)))
if mask.ndim > 2:
    mask = mask[...,0]

iSize = mask.shape[0]*mask.shape[1]
idx = np.nonzero(mask.reshape([iSize]) > 0)[0]
nPix = len(idx)
for a in range(nAngles):
    name = img_name.format(a+6)
    print(name)
    img = io.imread(os.path.join(root_dir,name))
    G = img[...,1]
    pix = G.reshape([iSize])[idx]
    vol[a] = np.mean(pix)

min = vol.min()
max = vol.max()
A = max-min
print("Slice amplitude: {}".format(A))

y = range(18)
plt.plot(y,vol)
plt.savefig('r6_nerve4.png')


mask_name = 'mask_nerve3.tif'
nAngles = 18
nImgs = 7
vol = np.zeros([nAngles])

mask = io.imread(os.path.join(root_dir,mask_name.format(0)))
if mask.ndim > 2:
    mask = mask[...,0]

iSize = mask.shape[0]*mask.shape[1]
idx = np.nonzero(mask.reshape([iSize]) > 0)[0]
nPix = len(idx)
for a in range(nAngles):
    name = img_name.format(a+6)
    print(name)
    img = io.imread(os.path.join(root_dir,name))
    G = img[...,1]
    pix = G.reshape([iSize])[idx]
    vol[a] = np.mean(pix)

min = vol.min()
max = vol.max()
A = max-min
print("Slice amplitude: {}".format(A))

y = range(18)
plt.plot(y,vol)
plt.savefig('r6_nerve3.png')

mask_name = 'mask_nerve2.tif'
nAngles = 18
nImgs = 7
vol = np.zeros([nAngles])

mask = io.imread(os.path.join(root_dir,mask_name.format(0)))
if mask.ndim > 2:
    mask = mask[...,0]

iSize = mask.shape[0]*mask.shape[1]
idx = np.nonzero(mask.reshape([iSize]) > 0)[0]
nPix = len(idx)
for a in range(nAngles):
    name = img_name.format(a+6)
    print(name)
    img = io.imread(os.path.join(root_dir,name))
    G = img[...,1]
    pix = G.reshape([iSize])[idx]
    vol[a] = np.mean(pix)

min = vol.min()
max = vol.max()
A = max-min
print("Slice amplitude: {}".format(A))

y = range(18)
plt.plot(y,vol)
plt.savefig('r6_nerve2.png')