import os
import sys
import fnmatch
import skimage.io as io
import tifffile
from misc.XMLUtils import XMLUtils
import logging
import glob
from misc.TiffTileLoader import TiffTileLoader
import numpy as np



def merge(coords_file, metadata_xml, tiles_dir):
    tiffLoader = TiffTileLoader()
    grid_rows, grid_cols, img_rows, img_cols, img_home, img_file = XMLUtils.parse_tiles_metadata(metadata_xml)
    out_file = '/home/maryana/storage/Posdoc/AVID/AV13/TEMP/MC1#460-Hippocampus/output/RES(14520x17736x1)/merged_image.tif'

    grid_tiles = (grid_rows,grid_cols)
    orig_size = (img_rows,img_cols)
    tiffLoader.merge_tiles_rgb(tiles_dir,coords_file,out_file,grid_tiles,orig_size)



def main():
    coords_file = '/home/maryana/storage/Posdoc/AVID/AV13/TEMP/MC1#460-Hippocampus/output/RES(14520x17736x1)/tiles/tile_coordinates.npy'
    xml_file = '/home/maryana/storage/Posdoc/AVID/AV13/TEMP/MC1#460-Hippocampus/output/RES(14520x17736x1)/tiles/tiling_info.xml'
    tiles_dir = '/home/maryana/storage/Posdoc/AVID/AV13/TEMP/MC1#460-Hippocampus/output/RES(14520x17736x1)/tiles'

    merge(coords_file,xml_file,tiles_dir)

if __name__ == '__main__':
    main()