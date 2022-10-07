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



input_dir = '/home/maryana/storage/Posdoc/Brains/AVID/AV13/blockface/seg'
output_dir = '/home/maryana/storage/Posdoc/Brains/AVID/AV13/blockface/seg/pad'
rows_pad = 1000 #each side will be padded with rows_pad new rows
cols_pad = 1000
value_pad = 0
file_ext = '.png'


def do_padding():
    files = glob.glob(os.path.join(input_dir, '*' + file_ext))
    for fPath in files:
        tind = [m.start() for m in re.finditer('/', fPath)]
        s = tind[-1]
        name = fPath[s+1:]
        print '   Processing file {F}'.format(F=name)
        name = os.path.splitext(name)[0]+file_ext
        newName = os.path.join(output_dir,name)
        img = io.imread(fPath)

        if img.ndim > 3:
            img = color.rgb2gray(img)

        if rows_pad > 0:
            rpad = np.ones([rows_pad, img.shape[1]]).astype('uint16') * value_pad
            img = np.concatenate((rpad,img,rpad),axis=0,)
        if cols_pad > 0:
            cpad = np.ones([img.shape[0],cols_pad]).astype('uint16') * value_pad
            img = np.concatenate((cpad,img,cpad),axis=1)

        io.imsave(newName,img)

def main():
    do_padding()

if  __name__ == '__main__':
    main()