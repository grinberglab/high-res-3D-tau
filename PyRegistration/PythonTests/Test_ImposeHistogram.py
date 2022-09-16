import glob
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage import color
import os
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform
from skimage.filters import gaussian

#case_dir = '/Volumes/SUSHI_HD2/SUSHI/Posdoc/PSP/P2724/slab-4/'
case_dir = '/home/maryana/storage/Posdoc/PSP/P2724/slab-4/'
imgPath1 = os.path.join(case_dir,'blockface/orig/P2724-4 _014.jpg')
imgPath2 = os.path.join(case_dir,'blockface/orig/P2724-4 _034.jpg')
img1 = io.imread(imgPath1) # ref histogram
img2 = io.imread(imgPath2) # histogram to be changed
img1 = xform.rescale(img1,0.20)
img2 = xform.rescale(img2,0.20)

# img1 = img_as_ubyte(color.rgb2gray(img1))
# img2 = img_as_ubyte(color.rgb2gray(img2))

R1 = img1[...,0]; G1 = img1[...,1]; B1 = img1[...,2]
R2 = img2[...,0]; G2 = img2[...,1]; B2 = img2[...,2]


def impose_hist(img, ref_hist):

    nTotalPix = img.shape[0]*img.shape[1]
    linImg = img.reshape([nTotalPix])
    sort_idx = np.argsort(linImg)

    newImg = -1 * np.ones(linImg.shape) # empty image vector
    maxVal = ref_hist.shape[0] # max gray value
    currPos = 0
    for currBin in range(0,maxVal):
        nPixInBin = ref_hist[currBin] #reference histogram
        for p in range(0,nPixInBin):
            if currPos > nTotalPix:
                print 'Warning: Index larger than num. image pixels'
                break
            origIdx = sort_idx[currPos]
            newImg[origIdx] = currBin # final image receives histogram bin value
            currPos += 1

    newImg[newImg < 0] = maxVal
    final_img = newImg.reshape(img.shape)
    return final_img


hR,bins = np.histogram(R1,bins=256,density=False)
Rf = impose_hist(R2,hR)
hG,bins = np.histogram(G1,bins=256,density=False)
Gf = impose_hist(G2,hG)
hB,bins = np.histogram(B1,bins=256,density=False)
Bf = impose_hist(B2,hB)

Rf = Rf.astype('uint8')
Gf = Gf.astype('uint8')
Bf = Bf.astype('uint8')
s = Rf.shape
final_img = np.concatenate((Rf.reshape([s[0], s[1],1]),Gf.reshape([s[0], s[1],1]),Bf.reshape([s[0], s[1],1])),axis=2)

plt.subplot(1,3,1)
plt.imshow(img1)
plt.subplot(1,3,2)
plt.imshow(final_img)
plt.subplot(1,3,3)
plt.imshow(img2)
plt.show()



