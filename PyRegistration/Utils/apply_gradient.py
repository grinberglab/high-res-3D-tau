import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import skimage.io as io
from skimage import color
import os
from skimage import img_as_float, img_as_ubyte
from skimage import transform as xform
import skimage.exposure as expo
from sklearn import linear_model


def estimate_grad(img):
    hsv = color.rgb2hsv(img)
    v = hsv[...,2]
    y = v[:,hsv.shape[1]/2]
    x = np.arange(y.shape[0])
    x = x.reshape(x.shape[0],1)
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    grad = np.matlib.repmat(y_pred, 1, img.shape[1])
    grad2 = grad[::-1, :]

    return grad2



def apply_grad_gamma(img_name):
    img = io.imread(img_name)

    if img.shape[0] < img.shape[1]:
        #img = xform.rotate(img,90)
        img = np.rot90(img,axes=(0,1))

    grad = estimate_grad(img)
    hsv = color.rgb2hsv(img)
    v = hsv[...,2]
    hsv2 = hsv.copy()
    v2 = grad*v
    hsv2[...,2] = v2
    img2 = color.hsv2rgb(hsv2)
    gam = expo.adjust_gamma(img2,gamma=1./2.,gain=1.5)
    gam[gam>1.0]=1.0
    gam[gam<0]=0
    gam2 = img_as_ubyte(gam)


    # img2 = img_as_float(img)
    # R = img2[..., 0].copy()
    # G = img2[..., 1].copy()
    # B = img2[..., 2].copy()
    # R2 = grad * R
    # G2 = grad * G
    # B2 = grad * B
    #
    # R3 = R2.astype('uint8')
    # G3 = G2.astype('uint8')
    # B3 = B2.astype('uint8')
    #
    # img3 = np.concatenate((R3.reshape(R3.shape[0], R3.shape[1], 1), G3.reshape(R3.shape[0], R3.shape[1], 1), B3.reshape(R3.shape[0], R3.shape[1], 1)), axis=2)
    # gam = expo.adjust_gamma(img3,gamma=1./2.,gain=1.5)
    # gam2 = img_as_ubyte(gam)

    return gam2

def main():

    #file_nums = [288]

    file_nums = [383,384,385,386,387,390,394,398,399,400,403,410,413,414,419,421,428,430,
                 348,349,350,351,352,353,363,364,366,368,369,370,371,376,377,378,379,380,381,
                 310,313,317,318,319,320,329,330,332,334,335,336,337,339,346,347,
                 286,287,288,290,291,293,294,295,296,297,298,303,304,306,308,309,
                 237,247,251,253,254,255,256,257,259,260,261,271,275,277,279,494,
                 190,193,195,199,206,208,214,215,216,217,232,233,235,478,483,489,
                 431,432,433,439,442,445,446,447,449,450,451,455,457,460,461,467,476,477,
                 607,610, 236, 246, 248, 249, 250, 262, 267, 268, 269, 276, 280, 281, 283, 284, 285,
                289, 300, 316, 321, 322, 324, 328, 331, 333, 340, 343, 344, 367, 395]

    #file_nums = np.arange(1,775)

    for file_num in file_nums:
        try:
            image_name = '/home/maryana/storage/Posdoc/AVID/AV23/blockface/orig/AV13-002_{:04d}.jpg'.format(file_num)
            out_name = '/home/maryana/storage/Posdoc/AVID/AV23/blockface/orig2/AV13-002_{:04d}.jpg'.format(file_num)
            img3 = apply_grad_gamma(image_name)
            plt.imshow(img3)
        except:
            print('Error processing file {}'.format(image_name))


        print('Saving {}.'.format(out_name))
        io.imsave(out_name,img3)



if __name__ == '__main__':
    main()