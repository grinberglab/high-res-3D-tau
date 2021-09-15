'''
Created on Dec 1, 2016

@author: maryana
'''


import fnmatch
import os
from skimage import io
import numpy as np
import sys
import glob

def prepare_images(img,pshape):
    if img.shape[0] > pshape[0]:
        img = img[0:pshape[0],...]
    if img.shape[1] > pshape[1]:
        img = img[:,0:pshape[1],:]

    return img


def compute_mean_image_RGB(root_dir,dest_file,binary_file,shape,file_ext='tif'):
            
    ncount = 0
    meanR = np.zeros(shape)
    meanG = np.zeros(shape)
    meanB = np.zeros(shape)

    nFiles = len(glob.glob(os.path.join(root_dir,'*.'+file_ext)))
    print('{} images to process.'.format(nFiles))

    for root, dir, files in os.walk(root_dir):
        for fname in fnmatch.filter(files, '*.'+file_ext):
            img_path = os.path.join(root,fname)
            try:
                img = io.imread(img_path)
                img = img.astype(float)
                img = prepare_images(img,shape)
            except:
                print('Error reading {}'.format(fname))
                continue
            
            R = img[:,:,0]
            G = img[:,:,1]
            B = img[:,:,2]
            
            #Based on Welford one-pass algorithm
            ncount+=1
            deltaR = R - meanR
            deltaG = G - meanG
            deltaB = B - meanB
            meanR += (deltaR/ncount)
            meanG += (deltaG/ncount)
            meanB += (deltaB/ncount)

            if ncount%1000 == 0:
                print('{} images processed.'.format(ncount))
            
                
    
    print('Total number of images processed: {}'.format(ncount))
    mR = meanR.reshape(meanR.shape[0],meanR.shape[1],1)
    mG = meanG.reshape(meanG.shape[0],meanG.shape[1],1)
    mB = meanB.reshape(meanB.shape[0],meanB.shape[1],1)
    mean_img = np.concatenate((mR,mG,mB),axis=2) #RGB
    #save mean image as numpy array
    np.save(dest_file,mean_img)






def main():

    if len(sys.argv) != 5:
        print('Usage: compute_mean_image <absolute_path_to_patches> <row size> <col size> <file_ext>')
        exit()

    root_dir = str(sys.argv[1]) #abs path to where the images are
    rshape = int(sys.argv[2]) #row size
    cshape = int(sys.argv[3]) #col size
    file_ext = str(sys.argv[4])
    pshape = [rshape,cshape]

    dest_file = os.path.join(root_dir,'mean_image.npy')
    binary_file = os.path.join(root_dir,'mean_image.binaryproto')
    #pshape = [256,256]

    
    print('Computing mean image from patches in: {}'.format(root_dir))
    compute_mean_image_RGB(root_dir, dest_file, binary_file, pshape, file_ext)
    print('Mean npy image saved in: {}'.format(dest_file))
    # print('Mean proto image saved in: {}'.format(binary_file))
    #
    
if __name__ == '__main__':
    main()
