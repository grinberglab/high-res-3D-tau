import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import glob
import os
import rawpy
import imageio
import re


root_dir = '/home/maryana/storage/Posdoc/PLI/2017/2372.10_Optic_Chiasm/PLI/raw'
dest_dir = '/home/maryana/storage/Posdoc/PLI/2017/2372.10_Optic_Chiasm/PLI/raw/TIFF'
#img_name = 'slice_0002_{:04d}.{}'
raw_ext = '.CR2'
tiff_ext = '.tif'

files = glob.glob(os.path.join(root_dir,'*'+raw_ext))
nFiles = len(files)
for fPath in files:
    tind = [m.start() for m in re.finditer('/', fPath)]
    s = tind[-1]
    name = fPath[s + 1:]
    print '   Processing file {F}'.format(F=name)
    name = os.path.splitext(name)[0] + tiff_ext
    newName = os.path.join(dest_dir,name)

    raw = rawpy.imread(fPath)
    img = raw.postprocess(output_bps=16)  # reads values in 16bits
    imageio.imsave(newName,img)

# vol = np.zeros([tmp.shape[0], tmp.shape[1], nFiles])
# for f in range(nFiles):
#     name = os.path.join(root_dir,img_name.format(f,raw_ext))
#     raw = rawpy.imread(name)
#     img = raw.postprocess(output_bps=16) #reads values in 16bits
#     G = img[...,1] #green channel
#     new_name = os.path.join(dest_dir,img_name.format(f,tiff_ext))
#     imageio.imsave(new_name,G)

