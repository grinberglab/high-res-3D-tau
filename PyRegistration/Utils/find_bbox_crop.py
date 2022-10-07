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

input_dir = '/Volumes/SUSHI_HD/SUSHI/Posdoc/PLI/2017/10181.10_Optic_Chiasm/blockface/seg_highres'
output_dir = '/Volumes/SUSHI_HD/SUSHI/Posdoc/PLI/2017/10181.10_Optic_Chiasm/blockface/seg_highres/crop'
file_ext = '.png'

def do_cropping():
    files = glob.glob(os.path.join(input_dir, '*' + file_ext))
    bound_box = np.array([-1,-1,-1,-1]) #[min_row, min_col, max_row, max_col]

    #find bound box
    for fPath in files:
        tind = [m.start() for m in re.finditer('/', fPath)]
        s = tind[-1]
        name = fPath[s+1:]
        print '   Reading file {F}'.format(F=name)
        img = io.imread(fPath)

        if img.ndim > 3:
            img = color.rgb2gray(img)

        mask = np.zeros(img.shape)
        mask[img > 0] = 255
        labels = meas.label(mask)
        props = meas.regionprops(labels)
        bbox = props[0].bbox #there should be only one object in the image

        if bound_box[0] == -1: # not initialized
            bound_box[:] = bbox
        if bbox[0] < bound_box[0]: #min row
            bound_box[0] = bbox[0]
        if bbox[1] < bound_box[1]: # min col
            bound_box[1] = bbox[1]
        if bbox[2] > bound_box[2]:  # max row
            bound_box[2] = bbox[2]
        if bbox[3] > bound_box[3]:  # max col
            bound_box[3] = bbox[3]

    #crop images
    for fPath in files:
        tind = [m.start() for m in re.finditer('/', fPath)]
        s = tind[-1]
        name = fPath[s + 1:]
        print '   Cropping file {F}'.format(F=name)
        name = os.path.splitext(name)[0] + file_ext
        newName = os.path.join(output_dir, name)
        img = io.imread(fPath)

        if img.ndim > 3:
            img = color.rgb2gray(img)

        img2 = img[bound_box[0]:bound_box[2],bound_box[1]:bound_box[3]]
        io.imsave(newName,img2)


def main():
    do_cropping()

if  __name__ == '__main__':
    main()