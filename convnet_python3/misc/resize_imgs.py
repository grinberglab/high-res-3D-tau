import matplotlib.pyplot as plt
import fnmatch
import os
from skimage import io
import numpy as np
import sys
import glob
import ntpath
import skimage.io as io
import skimage.transform as xform
from skimage import img_as_ubyte
from ImageUtil import compute_mean_image_RGB as compute_mean_image_RGB



def resize_images(root_dir,out_dir,ratio):
    files = glob.glob(os.path.join(root_dir, '*.tif'))
    nFiles = len(files) #assume tiles in different dirs have the same naming pattern
    for f in range(nFiles): #process one stack of tiles at a time
        file_path = files[f]
        file_name = ntpath.basename(file_path)

        img = io.imread(file_path)
        img2 = img_as_ubyte(xform.rescale(img,ratio))

        new_path = os.path.join(out_dir,file_name)
        io.imsave(new_path,img2)

def main():
    if len(sys.argv) != 4:
        print('Usage: resize_imgs <absolute_path_to_files_dir> <absolute_path_to_output_dir> <ratio>')
        exit()

    root_dir = str(sys.argv[1])  # abs path to where the images are
    out_dir = str(sys.argv[2])
    ratio = float(sys.argv[3])

    print('Images dir.: ' + root_dir)
    print('Output dir.: ' + out_dir)
    print('Resize ratio:' + str(ratio))

    resize_images(root_dir,out_dir,ratio)


if __name__ == '__main__':
    main()