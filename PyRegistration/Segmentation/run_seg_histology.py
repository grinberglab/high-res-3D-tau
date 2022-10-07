'''
Created on May 23, 2016

@author: Maryana Alegro
'''

import glob
from SegHistology import SegHistology

from SegHelper import SegHelper
import matplotlib.pyplot as plt
import numpy as np
import re
import skimage.io as io


def main():
    #imgPath = io.imread('/Users/maryana/Pictures/brainstem.jpg')
    #imgPath = '/Users/maryana/Pictures/brainstem.jpg'
    imgPath = '/home/maryana/storage/Posdoc/Brainstem/P2540/histology/orig/P2540.01.jpg'
    #imgPath2 = '/home/maryana/workspace-python/PyRegistration/P2540_048.jpg'
    
    root_dir = '/home/maryana/storage/Posdoc/Brainstem/P2540/'
    seg_dir = root_dir + 'histology/seg/'
    mask_dir = seg_dir + 'mask/'
    orig_dir = root_dir + 'histology/orig/'
    
    files = glob.glob(orig_dir+'*.jpg')
    #files = [orig_dir+'P2540_130.jpg']
    nFiles = len(files)
    
    sF,sB = SegHelper.getSamplesLAB(imgPath)
    segHisto = SegHistology(sF,sB)
    print segHisto.pipelineStage
    print '   {NF} file(s) found.'.format(NF=nFiles)
    #segHisto = SegSlides(sF,sB)
    #mask,img = segHisto.doSegmentation(imgPath)
    for fPath in files:
        tind = [m.start() for m in re.finditer('/', fPath)]
        s = tind[-1]
         
        name = fPath[s+1:]
        print '   Processing file {F}'.format(F=name)
         
        newName = seg_dir + name
        newMaskName = mask_dir + 'mask_' + name
         
        mask,img = segHisto.doLABSegmentation(fPath)
        mask = mask.astype(np.ubyte)
         
        io.imsave(newName,img)
        io.imsave(newMaskName,mask)
        
    print 'Finished.'
    
if __name__ == '__main__':
    main()