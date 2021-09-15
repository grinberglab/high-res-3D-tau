import sys
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import seg_background as seg_bkg
import glob
import os


def get_histo_files(img_dir):
    list_files = glob.glob(os.path.join(img_dir,'*.tif'))
    return list_files


def run_batch_segmentation(in_dir,out_dir):
    files = get_histo_files(in_dir)
    for histo_name in files:
        file_name = os.path.basename(histo_name)
        base_name = os.path.splitext(file_name)[0]
        mask_name = os.path.join(out_dir,base_name+'_background_mask.tif')
        seg_bkg.run_seg_background(histo_name,[],mask_name)


def main():

    if len(sys.argv) != 3:
        print('Usage: run_seg_background <rescaled histo_dir> <mask_dir>')
        print('Example: run_seg_background /AVID/AV13/AT100#440/res10/img /AVID/AV13/AT100#440/res10/brain_mask')
        exit()

    in_dir = str(sys.argv[1])
    out_dir = str(sys.argv[2])

    # img = io.imread(imgname)
    # seg_background(img, segname, outname)
    run_batch_segmentation(in_dir,out_dir)


if __name__ == '__main__':
    main()
