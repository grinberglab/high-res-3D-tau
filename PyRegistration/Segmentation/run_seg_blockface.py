'''
Created on May 23, 2016

@author: Maryana Alegro
'''

import matplotlib as mpl
mpl.use('tkagg')
import glob
from SegBlockface import SegBlockface
from SegHelper import SegHelper
import matplotlib.pyplot as plt
import numpy as np
import re
import skimage.io as io
from skimage import color
import os
import sys
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform


def main():

    if len(sys.argv) != 4:
        print('Usage: run_seg_blockface <imgs_dir> <ref_img> <do_ncut = 0|1>')
        exit()

    root_dir = str(sys.argv[1])
    imgPath = str(sys.argv[2])
    do_ncut = False
    if int(sys.argv[3]) > 0 :
        do_ncut = True

    files=[]


    seg_dir = os.path.join(root_dir, 'seg/')
    mask_dir = os.path.join(seg_dir,'mask/')
    orig_dir = os.path.join(root_dir,'orig/')

    if not files:
        files = glob.glob(orig_dir+'*.JPG')

    nFiles = len(files)
    
    sF,sB,idx_sF,idx_sB = SegHelper.getSamplesLAB(imgPath)
    ref_hist = SegHelper.getRefHistogram(imgPath)
    segBlock = SegBlockface(sF,sB,idx_sF,idx_sB,ref_hist)
    
    print (segBlock.pipelineStage)
    print ('   {NF} file(s) found.'.format(NF=nFiles))

    for fPath in files:
        tind = [m.start() for m in re.finditer('/', fPath)]
        s = tind[-1]
        name = fPath[s+1:]
        print ('   Processing file {F}'.format(F=name))
        name = os.path.splitext(name)[0]+'.png'
        newName = seg_dir + name
        newMaskName = mask_dir + 'mask_' + name

        mask,img,img_orig = segBlock.doSegmentation(fPath,do_ncut)
        img = color.rgb2gray(img)
        io.imsave(newName,img)
        io.imsave(newMaskName,mask)
        
    print ('Finished.')
    
if __name__ == '__main__':
    main()
