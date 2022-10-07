import skimage.io as io
import numpy as np
from skimage import transform as xform
from skimage import color
from skimage import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import glob
import os
import re

orig_dir = '/home/maryana/storage/Posdoc/PLI/2017/2372.10_Optic_Chiasm/blockface/orig'
seg_dir = '/home/maryana/storage/Posdoc/PLI/2017/2372.10_Optic_Chiasm/blockface/seg'
seg_dir2 = '/home/maryana/storage/Posdoc/PLI/2017/2372.10_Optic_Chiasm/blockface/seg_highres'


files = glob.glob(os.path.join(orig_dir,'*.jpg'))
nFiles = len(files)
print(nFiles)

for fPath in files:
    tind = [m.start() for m in re.finditer('/', fPath)]
    s = tind[-1]
    name = fPath[s + 1:]

    print(name)

    orig_name = os.path.join(orig_dir, name)
    name = os.path.splitext(name)[0] + '.png'
    seg_name = os.path.join(seg_dir, name)
    newName = os.path.join(seg_dir2, name)

    seg_img = io.imread(seg_name)
    mask = seg_img > 0

    orig_img = io.imread(orig_name)
    orig_img = img_as_ubyte(color.rgb2gray(orig_img))
    mask2 = xform.resize(mask, orig_img.shape)

    orig_img[mask2 <= 0] = 0

    io.imsave(newName, orig_img)