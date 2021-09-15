import os
import sys
import glob
from pipeline import export_heatmap_metadata as hmeta

#
# This script is supposed to run on the cluster
# Script for automatically tiling the full resolution histology images, using Image Magick


PIX_1MM = 819 #1mm= 819 pixels
PIX_5MM = 4095 #5mm = 4095 pixels

def get_files_dic(root_dir):
    case_dir_list = []
    list_dirs = glob.glob(root_dir + '/*/')
    for ldir in list_dirs:
        if os.path.isdir(ldir):
            if ldir.find('magick_tmp') != -1:
                continue
            case_dir_list.append(ldir)

    return case_dir_list


def main():
    if len(sys.argv) != 2:
        print('Usage: run_export_heatmap_metadata.py <root_dir>')
        exit()

    root_dir = str(sys.argv[1])  # abs path to where the images are
    case_dir_list = get_files_dic(root_dir)

    for di in case_dir_list:
        hmeta.export_metadata(di)


if __name__ == '__main__':
    main()