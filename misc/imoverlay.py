import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import skimage.io as io
from skimage import color
from skimage import exposure as exp
import mahotas
import os
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform

def imoverlay(img,mask,color=[1,1,1]):
    mask = mask.astype(bool)
    img = img_as_ubyte(img)

    if img.ndim == 3: #image is RGB
        R = img[...,0].copy()
        G = img[...,1].copy()
        B = img[...,2].copy()
    else:
        R = img.copy()
        G = img.copy()
        B = img.copy()

    R[mask] = 255*color[0]
    G[mask] = 255*color[1]
    B[mask] = 255*color[2]

    R_out = R.reshape(R.shape[0], R.shape[1], 1)
    G_out = G.reshape(G.shape[0], G.shape[1], 1)
    B_out = B.reshape(B.shape[0], B.shape[1], 1)

    out = np.concatenate((R_out, G_out, B_out), axis=2)
    return out
