'''
Created on Aug 3 2017

@author: maryana
'''

import matplotlib.pyplot as plt
import fnmatch
import os
from skimage import io
import numpy as np
import sys
import glob


def compute_mean_image_RGB(files):

    nFiles = len(files)

    img_size = np.array([])
    if nFiles > 0:
        img_name = files[0]
        img = io.imread(img_name)
        img_size = np.array([img.shape[0],img.shape[1]])
    else:
        return

    print('{} images to process.'.format(nFiles))

    ncount = 0
    meanR = np.zeros(img_size)
    meanG = np.zeros(img_size)
    meanB = np.zeros(img_size)
    for f in range(nFiles):
        img_name = files[f]
        try:
            img = io.imread(img_name)
            img = img.astype(float)
        except:
            print('Error reading {}'.format(img_name))
            continue

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        # Based on Welford one-pass algorithm
        ncount += 1
        deltaR = R - meanR
        deltaG = G - meanG
        deltaB = B - meanB
        meanR += (deltaR / ncount)
        meanG += (deltaG / ncount)
        meanB += (deltaB / ncount)

        if ncount % 1000 == 0:
            print('{} images processed.'.format(ncount))


    print('Total number of images processed: {}'.format(ncount))
    mR = meanR.reshape(meanR.shape[0],meanR.shape[1],1)
    mG = meanG.reshape(meanG.shape[0], meanG.shape[1], 1)
    mB = meanB.reshape(meanB.shape[0], meanB.shape[1], 1)
    mean_img = np.concatenate((mR, mG, mB), axis=2)  # RGB

    return mean_img

