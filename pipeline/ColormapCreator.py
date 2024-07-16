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
from lxml import etree as ET
import  export_heatmap_metadata  as exp_meta
import skimage.transform as xform
import nibabel as nib
from HeatmapCreator import  HeatmapCreator as hc
import matplotlib as mpl
import matplotlib.cm as cm
from skimage import img_as_ubyte

TILE_COORDS_FILE = 'tiles/tile_coordinates.npy' #inside output/RES???/, stores tiles coordinates
TILING_INFO_FILE = 'tiles/tiling_info.xml' #inside output/RES???/, stores gridsize and original file size
TILES_ADJ_METADATA = 'heat_map/TAU_seg_tiles/tiles_metadata.xml' # stores tiles adjacency information
TAU_SEG_DIR = 'heat_map/TAU_seg_tiles'
SEG_TILE_DIR = 'heat_map/seg_tiles'
HISTO_TILE_NAME = 'tile_{:04d}.tif'
SEG_TILE_NAME = 'tile_{:04d}_mask.tif'
SCALE_FILE = 'heat_map_{}_res10_scale.npy'
MIN_MAX_FILE = 'min_max.npy'
HMAP_RES = 0.1 # 0.1mm

class ColormapCreator(object):

    #Constructor
    def __init__(self,name,root_dir=None,dir_list=None):
        self.stage_name = name
        self.root_dir = root_dir
        self.dir_list = dir_list
        self.nErrors = 0
        self.config = None

        #init logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create a file handler
        log_name = os.path.join(root_dir,'ColormapCreator.log')
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(handler)

        #default values
        #self.PIX_1MM = 2890  # 1mm= 2890 pixels for 10x RO1
        #self.PIX_5MM = 14450  # 5mm = 14450 pixels for 10x RO1
        self.PIX_1MM = 5814  # 1mm= 5814 pixels for 20x Luke Syn
        self.PIX_5MM = 29070  # 5mm = 29070 pixels for 20x Luke Syn
        self.HMAP_RES = 0.1
        self.SCALE_FACTOR_VAL = 1000.0

        self.logger.info('Initializing stage.')




    def get_stage_name(self):
        return self.stage_name

    #run stage method
    def run_stage(self):
        if self.root_dir:

            slice_dirs = self.get_dirs_to_process(self.root_dir)

            self.logger.info('{}  directories to process.'.format(len(slice_dirs)))

            for slice_dir in slice_dirs:
                self.run_compute_colormap(slice_dir)

        else:
            self.logger.info('ROOT_DIR not set. Nothing to do.')
            self.nErrors += 1

        return self.nErrors


    def set_config(self,config):
        self.config = config
        if self.config:
            self.MEM_MAX = str(self.config.get('global', 'MAGICK_MEM'))
            self.PIX_1MM = int(self.config.get('global', 'PIX_1MM'))
            self.PIX_5MM = int(self.config.get('global', 'PIX_5MM'))
            self.HMAP_RES = float(self.config.get('heat_map', 'HMAP_RES'))
            self.SCALE_FACTOR_VAL = float(self.config.get('heat_map', 'SCALE_FACTOR'))

    def get_dirs_to_process(self,root_dir):
        dirs_list = []
        slice_dirs = glob.glob(os.path.join(root_dir, '*'))
        for sd in slice_dirs:
            if (os.path.isdir(sd) or os.path.islink(sd)) and sd.find('magick_tmp') == -1:
                dirs_list.append(sd)
        return dirs_list

    def compute_colormap(self,res10_file,colormap_file,scale_file,min_max_file,DEFAULT_SCALE_FACTOR = 1.0):

        if not os.path.exists(min_max_file):
            self.logger.info('MIN_MAX file does not exist. Stopping.')
            self.nErrors += 1
            return False

        if os.path.exists(scale_file):
            scale_factor = np.load(scale_file)[0]
        else:
            self.logger.info('SCALE_FACTOR file not found. Using default value.')
            scale_factor = DEFAULT_SCALE_FACTOR

        min_max = np.load(min_max_file)
        norm = mpl.colors.Normalize(vmin=min_max[0], vmax=min_max[1])
        cmap = cm.gray

        res10_map = np.load(res10_file)
        res10_map /= scale_factor #remove scale factor
        img = cmap(norm(res10_map)) #map "colors"
        img2 = img_as_ubyte(img)
        cmap = img2[:,:,0]

        #save colormap
        self.logger.info('Saving colormap TIFF file.')
        io.imsave(colormap_file, cmap)

        #save NIFTI
        self.logger.info('Saving colormap NIFTI file.')
        nii_name = colormap_file[0:-4] + '.nii'
        M = np.eye(4)
        nii = nib.Nifti1Image(cmap,M)
        nib.save(nii,nii_name)

        return True


    def run_compute_colormap(self,slice_dir):

        self.logger.info('*** Beginning to compute colormap {} ***'.format(slice_dir)) #i.e. /.../.../batch1/AT100_456/
        print('*** Beginning to compute colormap {} ***'.format(slice_dir))  # i.e. /.../.../batch1/AT100_456/

        hm_dir = os.path.join(slice_dir, 'heat_map/hm_map_' + str(self.HMAP_RES))
        res10_hm_file = os.path.join(hm_dir,'heat_map_'+str(self.HMAP_RES)+'_res10.npy')
        cmap_dir = os.path.join(slice_dir, 'heat_map/color_map_' + str(self.HMAP_RES))

        if not os.path.exists(cmap_dir):
            os.mkdir(cmap_dir)

        cmap_file = os.path.join(cmap_dir,'color_map_'+str(self.HMAP_RES)+'_res10.tif')
        scale_file = os.path.join(hm_dir,SCALE_FILE.format(str(self.HMAP_RES)))
        min_max_file = os.path.join(hm_dir,MIN_MAX_FILE)

        self.logger.debug('heatmap_dir: {}'.format(hm_dir))
        self.logger.debug('color_dir: {}'.format(cmap_dir))
        self.logger.debug('res10 image files: {}'.format(res10_hm_file))
        self.logger.debug('colormap file: {}'.format(cmap_file))
        self.logger.debug('min_max file: {}'.format(min_max_file))

        #compute colormap
        self.logger.info('Creating colormap.')
        print('Creating colormap.')
        status = self.compute_colormap(res10_hm_file, cmap_file, scale_file, min_max_file,self.SCALE_FACTOR_VAL)
        if status:
            self.logger.info('Colormap successfully created.')
        else:
            self.logger.info('There was ans error creating the colormap.')


