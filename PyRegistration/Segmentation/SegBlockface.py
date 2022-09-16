'''
Created on May 26, 2016

@author: maryana
'''

from SegSlides import SegSlides
import matplotlib.pyplot as plt
import numpy as np
from numpy import nonzero
from skimage import io
from skimage import color
from RectangleManager import RectangleManager
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform


class SegBlockface(SegSlides):
    '''
    classdocs
    '''

    def __init__(self, sF=[], sB = [], idx_sF=[], idx_sB=[], ref_hist=np.array([])):
        '''
        Constructor
        '''
        SegSlides.__init__(self,sF,sB,idx_sF,idx_sB,ref_hist)
        self.pipelineStage = 'Blockface Segmentation'
        
        
    def doSegmentation(self, imgPath, doNcut = False):
        img = io.imread(imgPath)
        img = xform.rescale(img,0.25)
        img = img_as_ubyte(img)
        
        return SegSlides.doSegmentation(self, img, run_ncut=doNcut)