import os
import sys
import fnmatch
import skimage.io as io
import logging
import glob
from PipelineRunner import  PipelineRunner
from ImageTiler import ImageTiler
from MaskTiler import MaskTiler
from TileMasker import TileMasker
from HeatmapCreator import HeatmapCreator
from ColormapCreator import ColormapCreator
import configparser


def main():
    if len(sys.argv) != 3:
        print('Usage: run_pipeline.py <root_dir> <config_file>')
        exit()

    root_dir = str(sys.argv[1])  # abs path to where the images are
    conf_file = str(sys.argv[2])

    print('### Creating pipeline ###')

    #create the pipeline
    pipeline = PipelineRunner(root_dir,conf_file)
    img_tiles = ImageTiler('Image Tiling',root_dir)
    mask_tiles = MaskTiler('Mask Resizing and Tiling',root_dir)
    apply_mask = TileMasker('Mask Tiles',root_dir)
    #heatmap_comp = HeatmapCreator('Compute HeatMap',root_dir)
    #colormap_comp = ColormapCreator('Compute ColorMap', root_dir)


    pipeline.add_stage(img_tiles)
    pipeline.add_stage(mask_tiles)
    pipeline.add_stage(apply_mask)
    #pipeline.add_stage(heatmap_comp)
    #pipeline.add_stage(colormap_comp)

    print('__________________________')
    print('Stages:')
    for st in pipeline.get_stages():
        print('    '+st.get_stage_name())
    print('__________________________')
    print('Starting...')

    #run pipeline
    pipeline.execute()

    print('### Pipeline Finished ###')





if __name__ == '__main__':
    main()
