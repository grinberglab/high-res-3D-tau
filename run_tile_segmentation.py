import os
import sys
import glob
from deprecated import prepare_tiles_4seg as prep_4seg
from convnet.deprecated import unet_segmentation as useg


def get_dir_dic(root_dir):
    dir_dic = {}

    list_dirs = glob.glob(root_dir + '/*/')
    for ldir in list_dirs:
        if os.path.isdir(ldir):
            if ldir.find('magick_tmp') != -1:
                continue

            # get mask tiles dir path
            tiles_dir = os.path.join(ldir, 'heat_map/seg_tiles')
            seg_dir = os.path.join(ldir, 'heat_map/TAU_seg_tiles')
            tmp_dir = os.path.join(ldir, 'heat_map/hdf5_tiles')

            if not os.path.exists(seg_dir):
                os.mkdir(seg_dir)
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)

            dir_dic[tiles_dir] = (seg_dir,tmp_dir)

    return dir_dic

def run_segmentation(root_dir,config_file):

    dirs_dic = get_dir_dic(root_dir)
    for di in dirs_dic.keys():
        dirs = dirs_dic[di]
        seg_dir = dirs[0]
        hd5_dir = dirs[1]

        print('Preparing tiles in {} for segmentation.'.format(di))
        #export temporary hdf5 for segmentation
        prep_4seg.run_prepare(di,hd5_dir,seg_dir,'tif')

        print('**************')
        print('** Beginning segmentation. It will take several minutes.')
        print('**************')
        #run convnet segmentation
        useg.run_segmentation(hd5_dir,seg_dir,config_file)

        status_file = os.path.join(seg_dir,'seg_complete.log')
        with open(status_file, 'w+') as sfile:
            sfile.write('1')




def main():
    if len(sys.argv) != 3:
        print('Usage: run_tile_segmentation <root_dir> <config_file>')
        exit()

    root_dir = str(sys.argv[1])  # abs path to where the images are
    config_file = str(sys.argv[2])  # abs path to where the images are
    run_segmentation(root_dir,config_file)


if __name__ == '__main__':
    main()

