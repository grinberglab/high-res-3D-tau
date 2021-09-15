
import os
import h5py
import numpy as np
from PIL import Image
import skimage.io as io
import sys
import matplotlib.pyplot as plt
import sklearn.feature_extraction as fx
from skimage import img_as_ubyte
import glob
import stridded_patches as sp
import cv2

NPIX_BKG = 0.98
S = 16
F = 48


def create_dataset(original_imgs_dir, mask_img_dir, dataset_img_dir, dataset_mask_dir, patch_size, stride, resize_mask = []):

    F = patch_size
    S = stride

    files = glob.glob(os.path.join(original_imgs_dir,'*.tif'))
    nImgs = len(files)
    if nImgs == 0:
        print("Images folder is empty")
        exit()

    if not os.path.exists(dataset_img_dir):
        os.mkdir(dataset_img_dir)
    if not os.path.exists(dataset_mask_dir):
        os.mkdir(dataset_mask_dir)

    patchCount = 0
    #for path, subdirs, files in os.walk(original_imgs_dir): #list all files, directories in the path
    for i in range(len(files)):
        basename = os.path.basename(files[i])
        img_name = os.path.join(original_imgs_dir, basename)
        ext = img_name[-3:]
        if ext != 'tif':
            continue

        print("original image: " + basename)

        mask_name = basename[0:-4] + "_mask.tif"
        print("ground truth name: " + mask_name)
        mask_name = os.path.join(mask_img_dir,mask_name)

        if not os.path.exists(mask_name):
            print("Mask doesn't exist. Skipping...")
            continue

        img = io.imread(img_name)
        g_truth = io.imread(mask_name)

        R = img[...,0]
        G = img[...,1]
        B = img[...,2]

        # process mask
        g_truth = img_as_ubyte(g_truth)
        mask_bkg = g_truth[..., 2] == 10
        mask_gm = g_truth[..., 2] == 255
        mask_thread = g_truth[..., 2] == 130
        mask_cell = g_truth[..., 2] == 0

        #mask_cell = mask_cell * 255
        #mask_thread = mask_thread * 255
        mask_fore = (mask_cell | mask_thread) * 255
        mask_bkg = (mask_gm | mask_bkg) * 255  # black background and GM together

        # img_patches = fx.image.extract_patches_2d(img, (y_dim, x_dim))
        # mask_patches = fx.image.extract_patches_2d(mask, (y_dim, x_dim))

        Rx = sp.get_stridded_view_square(R, F, S)
        Gx = sp.get_stridded_view_square(G, F, S)
        Bx = sp.get_stridded_view_square(B, F, S)

        #MCx = sp.get_strided_view_square(mask_cell, F, S)
        #MTx = sp.get_strided_view_square(mask_thread, F, S)
        MTx = sp.get_stridded_view_square(mask_fore, F, S)
        MBx = sp.get_stridded_view_square(mask_bkg, F, S)


        nPatches = Rx.shape[0]

        img_name_str = 'patch_{}_{}.tif'
        mask_name_str = 'patch_mask_{}_{}.npy'

        print('Number of patches: {}'.format(nPatches**2))

        for i in range(nPatches):
            for j in range(nPatches):

                is_bkg = 0

                r = Rx[i,j,...]
                g = Gx[i,j,...]
                b = Bx[i,j,...]
                img_patch = np.concatenate((r[...,np.newaxis],g[...,np.newaxis],b[...,np.newaxis]),axis=2).astype('uint8')

                #mmc = MCx[i,j,...]
                #mmt = MTx[i,j,...]
                #mmb = MBx[i,j,...]
                #mask_patch = np.concatenate((mmc[...,np.newaxis],mmt[...,np.newaxis],mmb[...,np.newaxis]),axis=2).astype('uint8')
                mmt = MTx[i,j,...]
                mmb = MBx[i,j,...]
                mask_patch = np.concatenate((mmt[...,np.newaxis],mmb[...,np.newaxis]),axis=2).astype('uint8')

                if resize_mask != []:
                    mask_patch = cv2.resize(mask_patch,resize_mask,interpolation=cv2.INTER_NEAREST)

                bkg = mask_patch[...,1]>0
                npx_bkg = len(np.nonzero(bkg)[0])
                npx_patch = bkg.shape[0]*bkg.shape[1]
                pcent = float(npx_bkg)/float(npx_patch)
                if pcent >= NPIX_BKG: #check if it's background patch
                    is_bkg = 1

                img_name = os.path.join(dataset_img_dir, img_name_str.format(is_bkg, patchCount))
                mask_name = os.path.join(dataset_mask_dir, mask_name_str.format(is_bkg, patchCount))

                io.imsave(img_name, img_patch)
                np.save(mask_name, mask_patch)

                patchCount += 1



def main():

    if len(sys.argv) != 7 and len(sys.argv) != 8:
        print('Usage: prepare_datasets_taunet_2classes <absolute_path_to_imgs> <absolute_path_to_masks> <dataset_path_images> '
              '<dataset_path_masks> <patch_size> <stride> [<mask_size>]')
        exit()

    original_imgs_dir = str(sys.argv[1])
    mask_imgs_dir = str(sys.argv[2]) #row size
    dataset_img_dir = str(sys.argv[3])
    dataset_mask_dir = str(sys.argv[4])
    patch_size = int(sys.argv[5])
    stride = int(sys.argv[6])
    if len(sys.argv) == 8:
        new_size = int(sys.argv[7])
        resize_mask = (new_size,new_size)
    else:
        resize_mask = []

    create_dataset(original_imgs_dir, mask_imgs_dir, dataset_img_dir, dataset_mask_dir, patch_size, stride, resize_mask)



if __name__ == '__main__':
    main()