
import sys
import glob
import os
import random
import shutil


def get_debug_pairs(img_dir,mask_dir,dest_dir,dest_mask_dir,num_pat):

    list_imgs = glob.glob(os.path.join(img_dir,'*_0_*.tif'))
    nImgs = len(list_imgs)

    if num_pat > nImgs:
        print('Error: number of debug patches must be less or equal to {}'.format(nImgs))
        exit()

    random.shuffle(list_imgs)
    for i in range(num_pat):
        file_name = list_imgs[i]
        basename = os.path.basename(file_name)
        mask_name = 'patch_mask_' + basename[6:-4] + '.npy'
        orig_mask_name = os.path.join(mask_dir, mask_name)

        try:
            shutil.copy(file_name,dest_dir)
        except Exception as e:
            print('Error copying file {}'.format(file_name))
            print(str(e))

        try:
            shutil.copy(orig_mask_name, dest_mask_dir)
        except Exception as e:
            print('Error copying file {}'.format(orig_mask_name))
            print(str(e))


def main():

    if len(sys.argv) != 6:
        print("Usage: create_debug_dataset <img_dir> <mask_dir> <debug_img_dir> <debug_mask_dir> <num. patches>")
        exit()

    img_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    dest_img_dir = sys.argv[3]
    dest_mask_dir = sys.argv[4]
    num_pat = int(sys.argv[5])

    get_debug_pairs(img_dir,mask_dir,dest_img_dir,dest_mask_dir,num_pat)


if __name__ == '__main__':
    main()