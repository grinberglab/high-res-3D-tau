import glob
from Segmentation.SegBlockface import SegBlockface
from Segmentation import SegHelper
from Segmentation import SegSlides
import matplotlib.pyplot as plt
import numpy as np
import re
import skimage.io as io
from skimage import color
import os
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform
from sklearn.cluster import KMeans
from skimage.filters import gaussian

sigma = 5
img_file = '/home/maryana/storage/Posdoc/WholeBrainBr/blockface/orig/12905.16-Whole Brain_141.jpg'
img_file2 = '/home/maryana/storage/Posdoc/WholeBrainBr/blockface/orig/12905.16-Whole Brain_541.jpg'
img = io.imread(img_file)
img = xform.rescale(img,0.20)
img = gaussian(img,sigma)

img2 = io.imread(img_file2)
img2 = xform.rescale(img2,0.20)
img2 = gaussian(img2,sigma)

#img = img_as_ubyte(img)
#img2 = img_as_ubyte(img2)

sB,sF = SegHelper.SegHelper.showImgGetSelection(img)
sB = np.round(sB).astype(int)
sF = np.round(sF).astype(int)
ref_hist = SegHelper.SegHelper.getRefHistogram(img_file)
segSlides = SegSlides.SegSlides(sF,sB,ref_hist)
lab = color.rgb2lab(img)
back = lab[sB[1]:sB[3], sB[0]:sB[2]]
fore = lab[sF[1]:sF[3], sF[0]:sF[2]]
mLf = np.mean(np.ravel(fore[..., 0]))
mAf = np.mean(np.ravel(fore[..., 1]))
mBf = np.mean(np.ravel(fore[..., 2]))
mLb = np.mean(np.ravel(back[..., 0]))
mAb = np.mean(np.ravel(back[..., 1]))
mBb = np.mean(np.ravel(back[..., 2]))
nPix = lab.shape[0]*lab.shape[1]
L = lab[...,0]
A = lab[...,1]
B = lab[...,2]
data = np.concatenate((L.reshape([nPix,1]),A.reshape([nPix,1]),B.reshape([nPix,1])),axis=1)
init = np.array([[mLf,mAf,mBf],[mLb,mAb,mBb]])
kmeans = KMeans(n_clusters=2, random_state=0,init=init).fit(data)
clusters = kmeans.predict(data)
mask = clusters.reshape([lab.shape[0],lab.shape[1]])


#img2 = segSlides.impose_histogram(img2)
lab2 = color.rgb2lab(img2)
nPix2 = lab2.shape[0]*lab2.shape[1]
L2 = lab2[...,0]
A2 = lab2[...,1]
B2 = lab2[...,2]
data2 = np.concatenate((L2.reshape([nPix2,1]),A2.reshape([nPix2,1]),B2.reshape([nPix2,1])),axis=1)
kmeans2 = KMeans(n_clusters=2, random_state=0,init=init).fit(data2)
clusters2 = kmeans2.predict(data2)
mask2 = clusters2.reshape([lab2.shape[0],lab2.shape[1]])
plt.imshow(mask2)






