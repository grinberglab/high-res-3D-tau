import cv2

import fnmatch
import numpy as np
import os
import random
import scipy.misc
import sys
import matplotlib.pyplot as plt
import skimage.transform as xform
from misc.XMLUtils import XMLUtils
from misc.TiffTileLoader import TiffTileLoader
import skimage.io as io
import tifffile
import glob
import uuid
import re

#Created by Riyana
#Edited by Maryana

THRESH=0.20


def get_dirs_to_process(root_dir):
    dirs_list = []
    slice_dirs = glob.glob(os.path.join(root_dir, '*'))
    for sd in slice_dirs:
        if (os.path.isdir(sd) or os.path.islink(sd)) and sd.find('magick_tmp') == -1:
            dirs_list.append(sd)
    return dirs_list

def get_files_info(root_dir):
    dirs = glob.glob(os.path.join(root_dir,'*'))

    print('Globed dirs: ')
    print(dirs)

    file_dic = {}
    for d in dirs:

        print(d)

        mask_tiles_dir = os.path.join(d,'mask/final_mask/tiles')
        seg_tiles_dir = os.path.join(d,'heat_map/seg_tiles')
        patch_mask_dir = os.path.join(d,'mask/patches_mask')

        print(mask_tiles_dir)
        print(seg_tiles_dir)
        print(patch_mask_dir)

        #find RES* folder
        output_dir = os.path.join(d,'output')
        res_dir = ''
        for root, dir, files in os.walk(output_dir):
            if fnmatch.fnmatch(root,'*/RES(*'): #it's inside /RES*
                res_dir = root
                break
        histo_tiles_dir = os.path.join(res_dir,'tiles')
        metadata_xml = os.path.join(histo_tiles_dir,'tiling_info.xml')

        #get patches mask
        patch_mask = None
        if os.path.exists(patch_mask_dir):
            files = glob.glob(os.path.join(patch_mask_dir,'*.tif'))
            if files:
                patch_mask = files[0] # there should be only one
            # if not os.path.exists(patch_mask):
            #     patch_mask = None


        if os.path.exists(mask_tiles_dir) and os.path.exists(seg_tiles_dir) and os.path.exists(metadata_xml):
            file_dic[d] = {'mask_tiles':mask_tiles_dir, 'seg_tiles':seg_tiles_dir, 'patch_mask':patch_mask, 'xml_file':metadata_xml}

    return file_dic

def print_dirs_info(dir_dic):

    for d in dir_dic.keys():
        dc = dir_dic[d]
        print(d)
        print(dc['mask_tiles'])
        print(dc['seg_tiles'])
        print(dc['patch_mask'])
        print(dc['xml_file'])


def calc_percentage(img_arr):
    print(np.count_nonzero(img_arr))
    print(np.count_nonzero(img_arr==0))
    print(np.prod(img_arr.shape))
    white_matter_percentage = np.count_nonzero(img_arr)/(float(np.prod(img_arr.shape)))
    print('white matter percentage: ' + str(white_matter_percentage))

    return white_matter_percentage


def get_gray_matter(img_arr):
    nonzero = np.nonzero(img_arr)
    coordinates = [(nonzero[0][i], nonzero[1][i]) for i in range(len(nonzero[0]))]
    return coordinates

def get_num_white(block):
    # num. non zeros in the blue channel
    tmp_nnz_b = block.flatten().nonzero()
    nnz_b = float(len(tmp_nnz_b[0]))  # number of non-zero pixel in BLOCK matrix
    return nnz_b


def collect_samples(root_dir, x_len, y_len, patch_count, hdir):

    #home_dir = os.getcwd()
    home_dir = hdir

    print('Home dir: {}'.format(home_dir))

    x = int(x_len)
    y = int(y_len)

    dirs_list = get_files_info(root_dir)
    print_dirs_info(dirs_list)

    print('Beginning to extract patches.')

    for sliceid in dirs_list.keys():
        print('Extracting patches from {}'.format(sliceid))
        slice_dic = dirs_list[sliceid]
        mask_tiles_dir = slice_dic['mask_tiles']
        seg_tiles_dir = slice_dic['seg_tiles']
        patch_mask = slice_dic['patch_mask']
        xml_file = slice_dic['xml_file']

        #fetch list of all mask tiles
        masked_file_list = glob.glob(os.path.join(mask_tiles_dir,'*.tif'))

        #fetch list of all histo tiles
        #colored_file_list = glob.glob(os.path.join(seg_tiles_dir,'*.tif'))


        if patch_mask:
            #read metadata
            grid_rows, grid_cols, img_rows, img_cols, img_home, img_file = XMLUtils.parse_tiles_metadata(xml_file)

            # read patches mask and compute virtual tile coords
            tiffLoader = TiffTileLoader()
            tiffLoader.open_file(patch_mask)
            tiffLoader.compute_tile_coords(grid_rows,grid_cols)


        #travel up to base directory after colored tiles traveral
        os.chdir(home_dir)
        print('dir after all tiles collected: ' + os.getcwd())

        print("size of file list = " + str(len(masked_file_list)))

        #parse through all tiles to extract patches
        count = 0
        #patch_count = 0
        for mask_file_name in masked_file_list:

            count+=1
            #os.chdir(masked_file_list[fn])

            #get tile number from file name
            filename = os.path.basename(mask_file_name) #tile names are always 'tile_????.tif'
            idx1 = filename.find('_')
            idx2 = filename.find('.')
            snum =  filename[idx1+1:idx2]
            snum = int(snum)

            colored_file_name = os.path.join(seg_tiles_dir,'tile_{:04d}.tif'.format(snum))

            #load tile
            tile_arr = cv2.imread(mask_file_name)
            if tile_arr.ndim > 1:
                tile_arr = tile_arr[...,0]
            #set minumum amount of pixels necessary in each tile
            tile_thresh = THRESH * tile_arr.shape[0] * tile_arr.shape[1]

            total_pixel_coords = get_num_white(tile_arr)

            if patch_mask:
                #load respective tile from patch mask
                tile_pmask_small = tiffLoader.get_tile_by_num(snum)
                if tile_pmask_small.ndim > 2:
                    tile_pmask_small = tile_pmask_small[...,0]
                #resize patch mask tile to match full res tile size
                tile_pmask = xform.resize(tile_pmask_small,tile_arr.shape,preserve_range=True).astype('uint8')

                tile_arr[tile_pmask < 10] = 0 #zero out pixels outside the ROI

            # get grey matter coordinates
            coordinates = get_gray_matter(tile_arr)

            if (len(coordinates) == 0) or (total_pixel_coords < tile_thresh):
                os.chdir(home_dir)
                continue

            print("calculated coordinates white matter")

            #determine how many samples are needed based on sampling logic
            needed_samples = 1

            #identifying patch centers
            patches = []
            patch_centers = []
            patch_points = []
            sample_counter = 0
            temp_coordinate = ()

            #avoid infinite loop
            max_tries = 10

            #make sure samples are the same size and do not exceed image boundaries
            while sample_counter < needed_samples and max_tries > 0:

                #calculate a random center and check if it is valid
                temp_coordinate = random.randint(0, len(coordinates) - 1)
                max_tries -= 1
                if (coordinates[temp_coordinate][0] - x/2) < 0 or \
                    (coordinates[temp_coordinate][1] - y/2) < 0 or \
                    (coordinates[temp_coordinate][0] + x/2) >= len(tile_arr) or \
                    (coordinates[temp_coordinate][1] + y/2) >= len(tile_arr[0]):
                    continue

                patch_centers.append(temp_coordinate)
                #patch_points.append([left, left, top, bottom])
                sample_counter += 1

            #begin patch extraction
            print("beginning patch extraction")
            os.chdir(home_dir)
            print('dir at start of patch extraction: ' + os.getcwd())

            if not os.path.exists('patches'):
                os.makedirs('patches')

            #extract square patches with center coordinate
            for i in range(len(patch_centers)):
                patch_x = coordinates[patch_centers[i]][0] - x/2
                patch_y = coordinates[patch_centers[i]][1] - y/2
                #patch = extract_color_patch(patch_x, x, patch_y, y, (count-1), colored_file_list)
                patch = extract_color_patch(patch_x, x, patch_y, y, colored_file_name)

                print('*** center: {}, x: {}, y: {}, patch_x: {}, patch_y: {}, tile: {}, count: {}'.format(patch_centers[i],
                                                            coordinates[patch_centers[i]][0],
                                                            coordinates[patch_centers[i]][1],
                                                            patch_x,patch_y,snum,count))

                os.chdir(home_dir)
                os.chdir('patches')

                #create UUID for file name
                uu_id = str(uuid.uuid1())

                #output_name = 'patch' + '_' + re.sub('[^A-Za-z0-9]+', '_', root_dir) + '_' + uu_id +'.tif'
                output_name = 'patch' + '_' + uu_id +'.tif'
                #scipy.misc.imsave('patch' + '_' + re.sub('[^A-Za-z0-9]+', '_', root_dir) + '_' + uu_id +'.tif', cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                cv2.imwrite(output_name,patch)
                #scipy.misc.imsave('patch' + '_' + str(patch_count)+'.tif', cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                patch_count += 1

            os.chdir(home_dir)


def extract_color_patch(patch_x, x, patch_y, y, colored_file_name):

    #os.chdir(colored_file_list[fn])
    print('dir during actual extraction: ' + os.getcwd())
    print('*** {}'.format(colored_file_name))
    colored_tile_arr = cv2.imread(colored_file_name)
    #tile_arr = np.array(file)
    print("extract color patch")
    #coordinates = get_white_matter(tile_arr)
    #needed_samples = int(calc_percentage(tile_arr) * 10)
    return colored_tile_arr[patch_x:(patch_x + x), patch_y:(patch_y + y)]



def main():
    #check for user input
    if (len(sys.argv) == 5):
        root_dir = sys.argv[1]
        x_len = sys.argv[2]
        y_len = sys.argv[3]
        hdir = str(sys.argv[4])
        print("collected arguments")

        patch_count = 0
        collect_samples(root_dir, x_len, y_len,patch_count,hdir)
    else:
        print("Usage: random_sampling_simple2.py <root_dir> <x_size> <y_size> <full_path_patches_dir>")
if __name__ == "__main__":
    main()