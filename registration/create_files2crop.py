import fnmatch
import os
from skimage import io
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from skimage import img_as_ubyte
import skimage.color as color
import nibabel as nib

block_dir = '/home/maryana/storage/Posdoc/AVID/AV13/blockface/nii/'
slices = ['504','296','440','457','472','488','536','552','600','632','648']
for slice_id in slices:
    nii_name=block_dir+'1181_001-Whole-Brain_0'+slice_id+'.png.nii'
    tiff_name=block_dir+'crop_'+slice_id+'.tif'
    nii = nib.load(nii_name)
    img = nii.get_data()
    io.imsave(tiff_name,img)
