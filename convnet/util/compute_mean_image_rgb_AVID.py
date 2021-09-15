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

TISSUE_THRESH = 0.90

def prepare_images(img,pshape):
    if img.shape[0] > pshape[0]:
        img = img[0:pshape[0],...]
    if img.shape[1] > pshape[1]:
        img = img[:,0:pshape[1],:]

    return img


def get_num_pix_tissue(img):  # assumes RGB image
        tmp_img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        tmp_nnz_b = tmp_img.flatten().nonzero()
        nnz_b = float(len(tmp_nnz_b[0]))  # number of non-zero pixel in img
        return nnz_b


def compute_mean_image_RGB(root_dir,dest_file,file_ext='tif'):
            
    ncount = 0
    nskip = 0
    meanR = []
    meanG = []
    meanB = []
    shape = []

    nFiles = len(glob.glob(os.path.join(root_dir,'*.'+file_ext)))
    print('{} images to process.'.format(nFiles))

    for root, dir, files in os.walk(root_dir):
        for fname in fnmatch.filter(files, '*.'+file_ext):
            img_path = os.path.join(root,fname)
            try:
                img = io.imread(img_path)

                if len(meanR) == 0:
                    shape = img.shape[0:2]
                    meanR = np.zeros(shape)
                    meanG = np.zeros(shape)
                    meanB = np.zeros(shape)
                    print('Size: {},{}'.format(shape[0],shape[1]))

                npix_tissue = get_num_pix_tissue(img)
                percent_tissue = npix_tissue / (img.shape[0]*img.shape[1])
                if percent_tissue < TISSUE_THRESH:
                    print('Image has too little tissue. Skipping. ({})'.format(percent_tissue))
                    nskip += 1
                    continue

                if img.shape[0] > shape[0] or img.shape[1] > shape[1]:
                    img = img[0:shape[0],0:shape[1],:]
                elif img.shape[0] < shape[0] or img.shape[1] < shape[1]:
                    print('Image is too small. Skipping.({},{})'.format(img.shape[0],img.shape[1]))
                    nskip += 1
                    continue

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
                print('{} images processed. {} skipped'.format(ncount,nskip))
            
                
    
    print('Total number of images processed: {}'.format(ncount))
    mR = meanR.reshape(meanR.shape[0],meanR.shape[1],1)
    mG = meanG.reshape(meanG.shape[0],meanG.shape[1],1)
    mB = meanB.reshape(meanB.shape[0],meanB.shape[1],1)
    mean_img = np.concatenate((mR,mG,mB),axis=2) #RGB
    #save mean image as numpy array
    np.save(dest_file,mean_img)
    print('{} images(s) processed.'.format((nFiles - nskip)))
    print('{} image(s) skipped.'.format(nskip))






def main():

    if len(sys.argv) != 3:
        #print('Usage: compute_mean_image_AVID <absolute_path_to_patches> <row size> <col size> <file_ext>')
        print( )
        exit()

    root_dir = str(sys.argv[1]) #abs path to where the images are
    #rshape = int(sys.argv[2]) #row size
    #cshape = int(sys.argv[3]) #col size
    file_ext = str(sys.argv[2])
    #pshape = [rshape,cshape]

    dest_file = os.path.join(root_dir,'mean_image.npy')
    #binary_file = os.path.join(root_dir,'mean_image.binaryproto')
    #pshape = [256,256]

    
    print('Computing mean image from patches in: {}'.format(root_dir))
    compute_mean_image_RGB(root_dir, dest_file, file_ext)
    print('Mean npy image saved in: {}'.format(dest_file))
    # print('Mean proto image saved in: {}'.format(binary_file))
    #
    
if __name__ == '__main__':
    main()
