import sys
import os
import glob
import numpy as np
import random


def split_dataset(orig_imgs_dir,mask_imgs_dir,data_img_dir,data_mask_dir,ptest):

    imgs_list = glob.glob(os.path.join(orig_imgs_dir,'*.tif'))
    nFiles = len(imgs_list)
    nTest = int(np.round(nFiles*ptest))

    #shuffle list
    random.shuffle(imgs_list)
    test_list = imgs_list[0:nTest]

    print('Creating data set...')
    for l in test_list:
        img_bname = os.path.basename(l)
        mask_bname = 'patch' + img_bname[5:-4] + '_mask.tif'

        img_name = os.path.join(orig_imgs_dir, img_bname)
        new_img_name = os.path.join(data_img_dir, img_bname)
        mask_name = os.path.join(mask_imgs_dir, mask_bname)
        new_mask_name = os.path.join(data_mask_dir, mask_bname)

        if not os.path.exists(img_name):
            print('Warning: image file {} does not exist. Skipping it.'.format(img_name))
            continue
        elif not os.path.exists(mask_name):
            print('Warning: mask file {} does not exist. Skipping it.'.format(mask_name))
            continue

        print('{} --> {}'.format(img_name, new_img_name))
        os.rename(img_name,new_img_name)
        print('{} --> {}'.format(mask_name, new_mask_name))
        os.rename(mask_name,new_mask_name)

def main():

    if len(sys.argv) != 6:
        print('Usage: split_datasets <absolute_path_to_imgs> <absolute_path_to_masks> <path_to_dataset_images>'
              ' <path_to_dataset_masks> <percentage [0,1]>')
        exit()

    original_imgs_dir = str(sys.argv[1])
    mask_imgs_dir = str(sys.argv[2])

    dataset_img_dir = str(sys.argv[3])
    dataset_mask_dir = str(sys.argv[4])
    percent = float(sys.argv[5])
    # percent_test = float(sys.argv[6])
    # percent_val = float(sys.argv[7])

    split_dataset(original_imgs_dir,mask_imgs_dir,dataset_img_dir,dataset_mask_dir,percent)



if __name__ == '__main__':
    main()