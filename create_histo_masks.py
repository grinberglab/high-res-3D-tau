###
# Fuses WM and Brain masks to create the final histology segmentation mask
###


import glob
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import mahotas as mht
import os
import ntpath
import sys
import skimage.color as color



MASK_VAL = 255

def create_masks(b_masks_dir,wm_masks_dir,outdir):
    #B_MASKS: folder where the histology img tiles are located
    #WM_MASKS: folder where the mask tiles are located
    #OUTDIR: folder where masked images are saved

    brain_masks = glob.glob(os.path.join(b_masks_dir,'*_background_mask.tif'))
    wm_masks = glob.glob(os.path.join(wm_masks_dir,'*_wm_mask.tif'))

    nMasks_b = len(brain_masks)
    nMasks_wm = len(wm_masks)

    if nMasks_b != nMasks_wm:
        print("Warning: the number of masks and images must match. ({}/{})".format(nMasks_b,nMasks_wm))
    else:
        print("Found " + str(nMasks_b) + " images to mask.")

    for f in range(nMasks_b):

        bmask_file = os.path.join(b_masks_dir,brain_masks[f])
        base_name = ntpath.basename(bmask_file)
        idx = base_name.find('_background_mask.tif')
        name_prefix = base_name[0:idx]
        wmmask_file = os.path.join(wm_masks_dir,name_prefix+'_wm_mask.tif')

        if not os.path.exists(wmmask_file):
            print("{} not found. Skipping.".format(wmmask_file))
            continue

        print("Joining {}, {}".format(bmask_file,wmmask_file))

        bmask = io.imread(bmask_file)
        if bmask.ndim > 2:
            if bmask.shape[2] >= 3:
                bmask = color.rgb2gray(bmask)
                bmask[bmask > 0] = 255
            else:
                bmask = bmask[...,0]
                bmask[bmask > 0] = 255
        wmmask = io.imread(wmmask_file)
        if wmmask.ndim > 2:
            if wmmask.shape[2] >= 3:
                wmmask = color.rgb2gray(wmmask)
                wmmask[wmmask > 0] = 255
            elif wmmask.shape[2] == 2:
                wmmask = wmmask[...,0]
                wmmask[wmmask > 0] = 255

        bmask[wmmask == MASK_VAL] = 0 #masks white matter out of the brain mask

        fmask_name = name_prefix+'_mask.tif'
        fmask_file = os.path.join(outdir,fmask_name)

        mht.imsave(fmask_file,bmask)
        print("File " + fmask_file + " saved.")


def main():

    if len(sys.argv) != 4:
        print('Usage: create_histo_masks <brain masks folder> <WM masks folder> <final masks folder>')
        #print('Example: seg_background /AVID/AV13/AT100#440/tiles AVID/AV13/AT100#440/tiles_mask AVID/AV13/AT100#440/tiles_seg')
        exit()

    braindir = str(sys.argv[1])
    wmdir = str(sys.argv[2])
    outdir = str(sys.argv[3])

    create_masks(braindir,wmdir,outdir)

    print('Images were successfully processed.')

if __name__ == '__main__':
    main()
