import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import rawpy
from scipy.optimize import curve_fit


root_dir = '/Volumes/SUSHI_HD2/SUSHI/Posdoc/PLI/2017/P2694/block-4/TIFF2'
img_name = 'slice_0002_{:04d}.tif'
nFiles = 18

tmp = io.imread(os.path.join(root_dir,img_name.format(0)))

vol = np.zeros([tmp.shape[0], tmp.shape[1], nFiles])
for f in range(nFiles):
    name = os.path.join(root_dir,img_name.format(f))
    img = io.imread(name)
    vol[...,f] = img


root_dir = '/Volumes/SUSHI_HD2/SUSHI/Posdoc/PLI/2017/P2694/block-4'
img_name = 'slice_0002_{:04d}.CR2'

vol2 = np.zeros([tmp.shape[0], tmp.shape[1], nFiles])
for f in range(nFiles):
    name = os.path.join(root_dir,img_name.format(f))
    raw = rawpy.imread(name)
    img = raw.postprocess(output_bps=16) #reads values in 16bits
    G = img[...,1] #green channel
    vol2[...,f] = G

d = vol-vol2
