import fnmatch
import os
from skimage import io
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cmppyh
from skimage import img_as_ubyte
import skimage.color as color
import nibabel as nib
import slice2nii as slice2nii



def get_histo_files(img_dir):
    list_files = glob.glob(os.path.join(img_dir,'*.tif'))
    return list_files


def run_segment_convert(res10_dir, masks_dir, out_dir, x_size, y_size, z_size):
    size = np.array([x_size, y_size, z_size, 0])
    list_histo = get_histo_files(res10_dir)
    for histo_file in list_histo:
        try:

            print('Processing {}'.format(histo_file))
            file_name = os.path.basename(histo_file)
            base_name = os.path.splitext(file_name)[0]
            mask_file = os.path.join(masks_dir,base_name+'_background_mask.tif')
            if not os.path.isfile(mask_file):
                print('Warning! No mask found for {}. Skipping!'.format(histo_file))
                continue

            img = io.imread(histo_file)
            if img.ndim > 2:
                img = color.rgb2gray(img)
            mask = io.imread(mask_file)
            if mask.ndim > 2:
                mask = mask[...,0]

            #mask image
            img[mask == 0] = 0

            #create nifti
            nii_name = os.path.join(out_dir,base_name+'.nii')
            slice2nii.save_nii(img,size,nii_name)
            print('File {} saved.'.format(nii_name))
        except Exception as e:
            print("Error processing {}".format(histo_file))
            print(e)


def main():
    if len(sys.argv) != 7:
        print('Usage: run_apply_masks_create_nii.py <res10_histo_dir> <final_masks_dir> <output_nii_dir> <x_dim> <y_dim> <z_dim>')
        exit()
    res10_dir = str(sys.argv[1])  # abs path to where the images are
    masks_dir = str(sys.argv[2])
    out_dir = str(sys.argv[3])
    x_size = float(sys.argv[4])
    y_size = float(sys.argv[5])
    z_size = float(sys.argv[6])
    run_segment_convert(res10_dir, masks_dir, out_dir, x_size, y_size, z_size)


if __name__ == '__main__':
    main()