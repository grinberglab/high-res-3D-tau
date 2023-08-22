# convert a 2d histology/block tiff into a nift file
# Import necessary libraries
import fnmatch
import os
from skimage import io
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from skimage import img_as_ubyte
import skimage.color as color
import nibabel as nib

# Function to save an image as a NIfTI file
def save_nii(img, size, nii_file):
    M = np.array([[size[0], 0, 0, 0],[0, size[1], 0, 0],[0, 0, size[2], 0],[0, 0, 0, 1]])
    if img.ndim > 2:
        if img.shape[2] == 3:
            img = color.rgb2gray(img)
        else:
            img = img[...,0]
    nii = nib.Nifti1Image(img, M)
    nib.save(nii, nii_file)

# Function to create a NIfTI object from an image
def create_nii(img, size):
    M = np.array([[size[0], 0, 0, 0],[0, size[1], 0, 0],[0, 0, size[2], 0],[0, 0, 0, 1]])
    if img.ndim > 2:
        if img.shape[2] == 3:
            img = color.rgb2gray(img)
        else:
            img = img[...,0]
    nii = nib.Nifti1Image(img, M)
    return nii

# Function to run the process of creating and saving a NIfTI file
def run_create_nii(img_file, x_size, y_size, z_size, nii_file):
    size = np.array([x_size, y_size, z_size, 0])
    img = io.imread(img_file)
    nii = create_nii(img, size)
    nib.save(nii, nii_file)

# Main function
def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 8:
        print('Usage: slice2nii.py <tiff_path> <x_size> <y_size> <z_size> <nifti_path> <start> <end>')
        exit()

    # Extract arguments from the command line
    img_file = str(sys.argv[1])  # Absolute path to where the images are
    x_size = float(sys.argv[2])
    y_size = float(sys.argv[3])
    z_size = float(sys.argv[4])
    nii_file = str(sys.argv[5])
    start = int(sys.argv[6])
    end = int(sys.argv[7])

    # Loop through the range of images
    for i in range(start, end + 1, 1):
        file_name = img_file + str(i) + ".tif"
        if os.path.isfile(file_name):
            print(i)
            nii_name = nii_file + str(i) + ".nii"
            run_create_nii(file_name, x_size, y_size, z_size, nii_name)

# Run the main function if the script is executed
if __name__ == '__main__':
    main()
