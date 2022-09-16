import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import skimage.io as io
from skimage import img_as_ubyte
import os

def pixel_total(W,F,S):
    F = float(F)
    W = float(W)
    S = float(S)
    o = F-S
    f = F - o
    w = np.ceil(W/S)-1
    s = w*f
    T = F + s
    return T

def compute_padding(W,F,S):
    if S >= F:
        return -1
    T = pixel_total(W,F,S)
    pad = T-W
    return pad


def compute_num_patch(W,S):
    return np.ceil(W/S)


def main():

    img = io.imread('/home/maryana/storage2/Posdoc/AVID/AV13/training/images/patch_320.tif')
    mask = io.imread('/home/maryana/storage2/Posdoc/AVID/AV13/training/masks/patch_320_mask.tif')
    out = '/home/maryana/storage2/Posdoc/AVID/AV13/training/tmp/img'
    out2 = '/home/maryana/storage2/Posdoc/AVID/AV13/training/tmp/mask'

    R = img[...,0]
    G = img[...,1]
    B = img[...,2]

    mask = img_as_ubyte(mask)
    m = mask[...,2]

    S = 26
    F = 100
    W = 1024

    pad = compute_padding(W,F,S)
    pad_cols = np.zeros((int(R.shape[0]+pad),int(pad)))
    pad_rows = np.zeros((int(pad),int(R.shape[1])))

    nPat = int(compute_num_patch(W,S))

    R1 = np.concatenate((R,pad_rows),axis=0)
    R2 = np.concatenate((R1,pad_cols),axis=1)

    G1 = np.concatenate((G,pad_rows),axis=0)
    G2 = np.concatenate((G1,pad_cols),axis=1)

    B1 = np.concatenate((B,pad_rows),axis=0)
    B2 = np.concatenate((B1,pad_cols),axis=1)

    m1 = np.concatenate((m,pad_rows),axis=0)
    m2 = np.concatenate((m1,pad_cols),axis=1)





    # mx = as_strided(m, shape=(1024-47,1024-47, 48, 48), strides=(m.strides[0],m.strides[1],m.strides[0],m.strides[1]))
    # Rx = as_strided(R, shape=(1,1009, 48, 48), strides=(R.strides[0],R.strides[1],R.strides[0],R.strides[1]))
    # Gx = as_strided(G, shape=(1,1009, 48, 48), strides=(G.strides[0],G.strides[1],G.strides[0],G.strides[1]))
    # Bx = as_strided(m, shape=(1,1009, 48, 48), strides=(B.strides[0],B.strides[1],B.strides[0],B.strides[1]))


    Rx = as_strided(R2, shape=(nPat,nPat, 48, 48), strides=(S*R2.strides[0],S*R2.strides[1],R2.strides[0],R2.strides[1]))
    Gx = as_strided(G2, shape=(nPat,nPat, 48, 48), strides=(S*G2.strides[0],S*G2.strides[1],G2.strides[0],G2.strides[1]))
    Bx = as_strided(B2, shape=(nPat,nPat, 48, 48), strides=(S*B2.strides[0],S*B2.strides[1],B2.strides[0],B2.strides[1]))
    mx = as_strided(m2, shape=(nPat,nPat, 48, 48), strides=(S*m2.strides[0],S*m2.strides[1],m2.strides[0],m2.strides[1]))

    for i in range(nPat):
        for j in range(nPat):
            r = Rx[i,j,...]
            g = Gx[i,j,...]
            b = Bx[i,j,...]
            rgb = np.concatenate((r[...,np.newaxis],g[...,np.newaxis],b[...,np.newaxis]),axis=2).astype('uint8')
            mm = mx[i,j,...]
            mm = mm.astype('uint8')

            out_rgb = os.path.join(out,'tile_{}_{}.tif'.format(i,j))
            out_m = os.path.join(out2,'tile_mask_{}_{}.tif'.format(i,j))

            io.imsave(out_rgb,rgb)
            io.imsave(out_m,mm)




if __name__ == '__main__':
    main()