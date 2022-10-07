'''
Created on May 24, 2016

@author: Maryana Alegro
'''

import matplotlib.pyplot as plt
import numpy as np
from numpy import nonzero
from skimage import io
from skimage import color
from RectangleManager import RectangleManager
from skimage import img_as_float,img_as_ubyte
from skimage import transform as xform

class SegHelper(object): 

    def __init__(self):
        self.fig, self.ax = plt.subplots()

    
    @staticmethod
    def getSamplesLAB(path):
        img = io.imread(path)
        img = img_as_float(img)
        img = xform.rescale(img,0.25,channel_axis = -1)
        
        #select samples
        #sB = sF = (x1,y1,x2,y2) 
        #x1,y1: upper left corner (y1: row, x1: col)
        #x2,y2: lower right corner (y2: row, x2: col)
        sB,sF = SegHelper.showImgGetSelection(img)
        sB = np.round(sB).astype(int)
        sF = np.round(sF).astype(int)
        idx_sB = sB
        idx_sF = sF

        lab = color.rgb2lab(img, channel_axis = -1)
        back = lab[sB[1]:sB[3],sB[0]:sB[2]]
        fore = lab[sF[1]:sF[3],sF[0]:sF[2]]
         
        mLf = np.mean(np.ravel(fore[...,0]))
        mAf = np.mean(np.ravel(fore[...,1]))
        mBf = np.mean(np.ravel(fore[...,2]))
         
        mLb = np.mean(np.ravel(back[...,0]))
        mAb = np.mean(np.ravel(back[...,1]))
        mBb = np.mean(np.ravel(back[...,2]))
         
        sF = (mLf,mAf,mBf)
        sB = (mLb,mAb,mBb)
         
        return sB,sF,idx_sB,idx_sF
        
    @staticmethod
    def showImgGetSelection(img):
        
        fig, ax = plt.subplots()
        plt.imshow(img)
        
        print ("Select the background and foreground sample pixels")
        print ("Press 'B' to store the BACKGROUND selection")
        print ("Press 'R' to store the FOREGROUND selection")
        print ("Close the window when done.")
        
        rectMng = RectangleManager(ax)
        plt.connect('key_press_event', rectMng.toggle_selector)
        fig.canvas.mpl_connect('key_press_event', rectMng.toggle_selector)
        plt.show()
        sB,sF = rectMng.getSelection()
        
        return sB,sF

    @staticmethod
    def getRefHistogram(imgPath):
        img_tmp = io.imread(imgPath)  # ref histogram
        img_tmp = xform.rescale(img_tmp, 0.25, channel_axis = -1)
        #img_tmp = img_as_ubyte(img_tmp)
        R = img_tmp[..., 0]
        G = img_tmp[..., 1]
        B = img_tmp[..., 2]
        hR, bins = np.histogram(R, bins=256, density=False)
        hG, bins = np.histogram(G, bins=256, density=False)
        hB, bins = np.histogram(B, bins=256, density=False)
        s = hR.shape
        hist = np.concatenate((hR.reshape([s[0],1]), hG.reshape([s[0],1]), hB.reshape([s[0],1])), axis=1)
        return hist

