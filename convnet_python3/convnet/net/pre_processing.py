import numpy as np
from PIL import Image
import cv2
import skimage.color as color
from help_functions import *


#My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = color.rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


def preproc_color(data,path_mu):
    mu = load_mean_values(path_mu)
    data = substract_mean(data,mu)
    data /= 255.
    #data = data/255.
    return data


def preproc_scale(data):

    maxR = np.max(data[:,0,...])
    maxG = np.max(data[:,1, ...])
    maxB = np.max(data[:,2, ...])

    minR = np.min(data[:,0,...])
    minG = np.min(data[:,1, ...])
    minB = np.min(data[:,2, ...])

    for im in range(data.shape[0]):
        R = data[im, 0, ...]
        G = data[im, 1, ...]
        B = data[im, 2, ...]

        R = (R - minR) / (maxR - minR)
        G = (G - minG) / (maxG - minG)
        B = (B - minB) / (maxB - minB)

        data[im, 0, ...] = R
        data[im, 1, ...] = G
        data[im, 2, ...] = B

    return data



def preproc_color2(data):

    meanR = np.mean(data[:,0,...])
    meanG = np.mean(data[:,1,...])
    meanB = np.mean(data[:,2,...])

    stdR = np.std(data[:,0,...])
    stdG = np.std(data[:,1, ...])
    stdB = np.std(data[:,2, ...])

    for im in range(data.shape[0]):
        R = data[im, 0, ...]
        G = data[im, 1, ...]
        B = data[im, 2, ...]

        R = (R - meanR) / stdR
        G = (G - meanG) / stdG
        B = (B - meanB) / stdB

        data[im, 0, ...] = R
        data[im, 1, ...] = G
        data[im, 2, ...] = B

    minR = np.min(data[:, 0, ...])
    minG = np.min(data[:, 1, ...])
    minB = np.min(data[:, 2, ...])

    maxR = np.max(data[:, 0, ...])
    maxG = np.max(data[:, 1, ...])
    maxB = np.max(data[:, 2, ...])

    for im in range(data.shape[0]):
        R = data[im, 0, ...]
        G = data[im, 1, ...]
        B = data[im, 2, ...]

        R = (R - minR) / (maxR - minR)
        G = (G - minG) / (maxG - minG)
        B = (B - minB) / (maxB - minB)

        data[im, 0, ...] = R
        data[im, 1, ...] = G
        data[im, 2, ...] = B

    return data


#============================================================
#========= PRE PROCESSING FUNCTIONS ========================#
#============================================================


def load_mean_values(path): #mean R,G,B
    #load mean values from pre-computed mean image
    mean_img = np.load(path)
    mr = mean_img[:,:,0]
    mg = mean_img[:,:,1]
    mb = mean_img[:,:,2]
    mu = np.array([mr.mean(),mg.mean(),mb.mean()])
    return mu #mu = [uR, uG, uB]

def substract_mean(data,mu):
    # mu = mu.reshape([1,3,1,1]) #reshape mu array to fit the dataset and allow substraction
    # imgs = imgs-mu

    muR = mu[0]
    muG = mu[1]
    muB = mu[2]

    for im in range(data.shape[0]):
        R = data[im, 0, ...]
        G = data[im, 1, ...]
        B = data[im, 2, ...]

        R = R - muR
        G = G - muG
        B = B - muB

        data[im, 0, ...] = R
        data[im, 1, ...] = G
        data[im, 2, ...] = B

    return data


#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs
