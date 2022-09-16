'''
Created on May 26, 2016

@author: maryana
'''
from SegSlides import SegSlides

from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from skimage import color
from skimage import segmentation as seg
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform
from skimage import morphology as morph
from skimage import filters as filters
from skimage import measure as meas
from skimage import feature as feat
from numpy import nonzero


class SegHistology(SegSlides):
    '''
    classdocs
    '''

    def __init__(self, sF=[], sB = []):
        '''
        Constructor
        '''
        SegSlides.__init__(self, sF, sB)
        self.pipelineStage = 'Histology Segmentation';
        
        
    
    def doLABSegmentation(self, imgPath):
        
        img = img_as_float(io.imread(imgPath));
        img = xform.rescale(img,0.25);
        imsize = img.shape[0:2];
        
        mask,img2 = SegSlides.doLABSegmentation(self, img);
        img2 = color.rgb2gray(img2);  
        
        cHull = morph.convex_hull_image(mask);
        cBorder = feat.canny(cHull);
        borderIdx = nonzero(cBorder == True);
        init = np.array([borderIdx[1], borderIdx[0]]).T
        
        snake = seg.active_contour(filters.gaussian(img2, 3),
                                   init, alpha=0.015, beta=10, gamma=0.001)
        

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        plt.gray()
        ax.imshow(img2)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img2.shape[1], img2.shape[0], 0])
        plt.show()
        
        pass
        