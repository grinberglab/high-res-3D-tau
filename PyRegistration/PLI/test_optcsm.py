import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import skimage.io as io
from skimage import color
import os
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform


root_dir = '/home/maryana/storage/Posdoc/PLI/2017/10181.10_Optic_Chiasm/PLI/raw/TIFF'
y = range(18)

img_name = 'slice_{:04d}_{:02d}.tif'
mask_name = 'mask_nerve4_slice_{:04d}_{:02d}.tif'
nAngles = 18
nImgs = 7
vol = np.zeros([nAngles,nImgs])

count = 0
for nSlice in range(15,22):
    mask = io.imread(os.path.join(root_dir,mask_name.format(nSlice,0)))
    if mask.ndim > 2:
        mask = mask[...,0]

    iSize = mask.shape[0]*mask.shape[1]
    idx = np.nonzero(mask.reshape([iSize]) > 0)[0]
    nPix = len(idx)
    for a in range(nAngles):
        name = img_name.format(nSlice,a)
        print(name)
        img = io.imread(os.path.join(root_dir,name))
        G = img[...,1]
        pix = G.reshape([iSize])[idx]
        vol[a,count] = np.mean(pix)
    count+=1

for n in range(nImgs):
    v = vol[:,n]
    min = v.min()
    max = v.max()
    A = max-min
    print("Slice {} amplitude: {}".format(n,A))

for n in range(nImgs):
    plt.plot(y,vol[:,n],label="{0}".format(n+15))
plt.legend(loc="upper left", bbox_to_anchor=[0, 1],ncol=2, shadow=True, title="Legend", fancybox=True)
plt.savefig('nerve4.png')

# img_name = 'slice_{:04d}_{:02d}.tif'
# mask_name = 'mask_nerve2_slice_{:04d}_{:02d}.tif'
# nAngles = 18
# nImgs = 4
# vol = np.zeros([nAngles,nImgs])
#
# count = 0
# for nSlice in range(25,29):
#     mask = io.imread(os.path.join(root_dir,mask_name.format(nSlice,0)))
#     if mask.ndim > 2:
#         mask = mask[...,0]
#
#     iSize = mask.shape[0]*mask.shape[1]
#     idx = np.nonzero(mask.reshape([iSize]) > 0)[0]
#     nPix = len(idx)
#     for a in range(nAngles):
#         name = img_name.format(nSlice,a)
#         print(name)
#         img = io.imread(os.path.join(root_dir,name))
#         G = img[...,1]
#         pix = G.reshape([iSize])[idx]
#         vol[a,count] = np.mean(pix)
#     count+=1
#
# for n in range(nImgs):
#     v = vol[:,n]
#     min = v.min()
#     max = v.max()
#     A = max-min
#     print("Slice {} amplitude: {}".format(n,A))
# for n in range(nImgs):
#     plt.plot(y,vol[:,n],label="{0}".format(n+25))
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],ncol=2, shadow=True, title="Legend", fancybox=True)
# plt.savefig('nerve2.png')
#
# img_name = 'slice_{:04d}_{:02d}.tif'
# mask_name = 'mask_nerve3_slice_{:04d}_{:02d}.tif'
# nAngles = 18
# nImgs = 7
# vol = np.zeros([nAngles,nImgs])
#
# count = 0
# for nSlice in range(15,22):
#     mask = io.imread(os.path.join(root_dir,mask_name.format(nSlice,0)))
#     if mask.ndim > 2:
#         mask = mask[...,0]
#
#     iSize = mask.shape[0]*mask.shape[1]
#     idx = np.nonzero(mask.reshape([iSize]) > 0)[0]
#     nPix = len(idx)
#     for a in range(nAngles):
#         name = img_name.format(nSlice,a)
#         print(name)
#         img = io.imread(os.path.join(root_dir,name))
#         G = img[...,1]
#         pix = G.reshape([iSize])[idx]
#         vol[a,count] = np.mean(pix)
#     count+=1
#
# for n in range(nImgs):
#     v = vol[:,n]
#     min = v.min()
#     max = v.max()
#     A = max-min
#     print("Slice {} amplitude: {}".format(n,A))
#
# for n in range(nImgs):
#     plt.plot(y,vol[:,n],label="{0}".format(n+15))
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],ncol=2, shadow=True, title="Legend", fancybox=True)
# plt.savefig('nerve3.png')
#
#
#
# img_name = 'slice_{:04d}_{:02d}.tif'
# mask_name = 'mask_nerve1_50_slice_{:04d}_{:02d}.tif'
# nAngles = 18
# nImgs = 6
# vol = np.zeros([nAngles,nImgs])
#
# count = 0
# for nSlice in range(1,7):
#     mask = io.imread(os.path.join(root_dir,mask_name.format(nSlice,0)))
#     if mask.ndim > 2:
#         mask = mask[...,0]
#
#     iSize = mask.shape[0]*mask.shape[1]
#     idx = np.nonzero(mask.reshape([iSize]) > 0)[0]
#     nPix = len(idx)
#     for a in range(nAngles):
#         name = img_name.format(nSlice,a)
#         print(name)
#         img = io.imread(os.path.join(root_dir,name))
#         G = img[...,1]
#         pix = G.reshape([iSize])[idx]
#         vol[a,count] = np.mean(pix)
#     count+=1
#
# for n in range(nImgs):
#     v = vol[:,n]
#     min = v.min()
#     max = v.max()
#     A = max-min
#     print("Slice {} amplitude: {}".format(n,A))
#
# for n in range(nImgs):
#     plt.plot(y,vol[:,n],label="{0}".format(n+15))
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],ncol=2, shadow=True, title="Legend", fancybox=True)
# plt.savefig('nerve1_50.png')
#
# img_name = 'slice_{:04d}_{:02d}.tif'
# mask_name = 'mask_nerve1_130_slice_{:04d}_{:02d}.tif'
# nAngles = 18
# nImgs = 6
# vol = np.zeros([nAngles,nImgs])
#
# count = 0
# for nSlice in range(1,7):
#     mask = io.imread(os.path.join(root_dir,mask_name.format(nSlice,0)))
#     if mask.ndim > 2:
#         mask = mask[...,0]
#
#     iSize = mask.shape[0]*mask.shape[1]
#     idx = np.nonzero(mask.reshape([iSize]) > 0)[0]
#     nPix = len(idx)
#     for a in range(nAngles):
#         name = img_name.format(nSlice,a)
#         print(name)
#         img = io.imread(os.path.join(root_dir,name))
#         G = img[...,1]
#         pix = G.reshape([iSize])[idx]
#         vol[a,count] = np.mean(pix)
#     count+=1
#
# for n in range(nImgs):
#     v = vol[:,n]
#     min = v.min()
#     max = v.max()
#     A = max-min
#     print("Slice {} amplitude: {}".format(n,A))
#
# for n in range(nImgs):
#     plt.plot(y,vol[:,n],label="{0}".format(n+15))
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],ncol=2, shadow=True, title="Legend", fancybox=True)
# plt.savefig('nerve1_130.png')