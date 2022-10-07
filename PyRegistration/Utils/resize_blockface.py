import glob
import re
import os
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from skimage import color
from skimage import img_as_float, img_as_ubyte
from skimage import morphology as morph
from skimage import filters as filters
from skimage import measure as meas
from numpy import nonzero
from skimage.measure import find_contours
from skimage.filters import gaussian
import scipy.ndimage.morphology as morph2
from skimage import transform as xform

#input_dir = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab+4/blockface/seg'
#input_dir = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab-2/blockface/seg'
#input_dir = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab-6/blockface/seg'
input_dir = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab-4/blockface/seg'
#output_dir = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab+4/blockface/seg/small'
#output_dir = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab-2/blockface/seg/small'
#output_dir = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab-6/blockface/seg/small'
output_dir = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab-4/blockface/seg/small'

files = glob.glob(os.path.join(input_dir,'*.png'))
for fPath in files:
    tind = [m.start() for m in re.finditer('/', fPath)]
    s = tind[-1]
    name = fPath[s+1:]
    print '   Processing file {F}'.format(F=name)
    name = os.path.splitext(name)[0]+'.png'
    newName = os.path.join(output_dir, name)

    img = io.imread(fPath)
    img2 = xform.rescale(img, 0.25)
    io.imsave(newName,img2)

