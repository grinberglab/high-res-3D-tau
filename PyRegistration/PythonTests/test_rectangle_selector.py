'''
Created on May 25, 2016

@author: Maryana Alegro
'''
from matplotlib import pyplot as plt
from skimage import color
from skimage import data
from skimage import img_as_float, img_as_ubyte
from skimage import io
from skimage import segmentation as seg
import skimage

from Segmentation.RectangleManager import RectangleManager
import numpy as np


def main():
    imgPath = '/home/maryana/Projects/workspace-python/PyRegistration/P2724-2_002.jpg'
    img = io.imread(imgPath)
    
    fig, current_ax = plt.subplots()
    plt.imshow(img)
    rectMng = RectangleManager(current_ax)
    plt.connect('key_press_event', rectMng.toggle_selector)
    plt.show()
    sB,sF = rectMng.getSelection()
    print sB
    print sF


if __name__ == '__main__':
    main()