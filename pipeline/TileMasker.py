import os
import sys
import fnmatch
import skimage.io as io
from misc.XMLUtils import XMLUtils
import glob
import mahotas as mht
import numpy as np
import logging
import configparser


class TileMasker(object):

    def __init__(self,name,root_dir):
        self.stage_name = name
        self.root_dir = root_dir
        self.nErrors = 0
        self.config = None

        #init logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create a file handler
        log_name = os.path.join(root_dir,'TileMasker.log')
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(handler)

        #Default values
        self.MASK_VAL = 255

    def set_config(self,config):
        self.config = config
        if self.config:
            self.MASK_VAL = int(self.config.get('tile_masking', 'MASK_VAL'))

    def get_stage_name(self):
        return self.stage_name

    def run_stage(self):
        # root_dir = '/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res'
        # root_dir= '/Users/maryana/Posdoc/AVID/AV13/TEMP'
        self.apply_masks(self.root_dir)

        return self.nErrors


    def apply_masks(self,root_dir):
        list_dirs = glob.glob(root_dir+'/*/')

        self.logger.debug('Directories: %s',list_dirs)

        for ldir in list_dirs:

            self.logger.debug('*** Processing %s ***', ldir)
            print('Processing {}'.format(ldir))

            if os.path.isdir(ldir):
                if ldir.find('magick_tmp') != -1:
                    continue

                #create segmented histo dir
                #self.create_dir_struct(os.path.join(root_dir,ldir))
                self.create_dir_struct(os.path.join(ldir))
                self.logger.info('Directory structure created.')

                #get mask tiles dir path
                mask_dir = os.path.join(ldir,'mask/final_mask/tiles')
                output_dir = os.path.join(ldir,'output')
                histo_dir = ''
                seg_dir = os.path.join(ldir,'heat_map/seg_tiles')

                #iterate inside OUTPUT_DIR to find histo tiles dir
                # we do this the RES(???x???) folder, whose name can change from slice to slice
                for root, dir, files in os.walk(output_dir):
                    if fnmatch.fnmatch(root, '*tiles'):  # it's inside /tiles*
                        histo_dir = root

                if not os.path.isdir(mask_dir) or not os.path.isdir(histo_dir):
                    print('{} or {} is not a directory'.format(mask_dir,histo_dir))

                tiles_mask = glob.glob(os.path.join(mask_dir,'*.tif'))
                tiles_histo = glob.glob(os.path.join(histo_dir,'*.tif'))
                nT_mask = len(tiles_mask)
                nT_histo = len(tiles_histo)

                if nT_mask != nT_histo:
                    self.logger.info('ERROR: the number of mask and histology tiles does not match. Skipping this images.')
                    print('ERROR: the number o mask and histology tiles is different. Skipping this images.')
                    continue

                for mTile in tiles_mask:
                    mask_tile_name = os.path.join(mask_dir,mTile)
                    base_name = os.path.basename(mask_tile_name)
                    histo_tile_name = os.path.join(histo_dir,base_name)

                    #check if files exist
                    if not os.path.exists(mask_tile_name):
                        self.logger.info('ERROR: file {} does not exist'.format(mask_tile_name))
                        print('Error: file {} does not exist'.format(mask_tile_name))
                        self.nErrors += 1
                        continue
                    if not os.path.exists(histo_tile_name):
                        self.logger.info('ERROR: file {} does not exist'.format(histo_tile_name))
                        print('Error: file {} does not exist'.format(histo_tile_name))
                        self.nErrors += 1
                        continue

                    print('Processing {}/{}'.format(histo_tile_name,mask_tile_name))
                    self.logger.info('Processing {}/{}'.format(histo_tile_name,mask_tile_name))

                    mask = io.imread(mask_tile_name)
                    img = io.imread(histo_tile_name)

                    if mask.ndim > 2:
                        mask = mask[:,:,0]

                    if not (mask.shape[0] == img.shape[0] and mask.shape[1] == img.shape[1]):
                        tmp = mht.imresize(mask, (img.shape[0], img.shape[1]))
                        tmp[tmp < 1] = 0
                        tmp[tmp > 0] = 255
                        mask = tmp.astype('ubyte')

                    size = img.shape[0:2]

                    R = img[..., 0].copy()
                    G = img[..., 1].copy()
                    B = img[..., 2].copy()

                    R[mask < self.MASK_VAL] = 0
                    G[mask < self.MASK_VAL] = 0
                    B[mask < self.MASK_VAL] = 0

                    R = R.reshape([size[0], size[1], 1])
                    G = G.reshape([size[0], size[1], 1])
                    B = B.reshape([size[0], size[1], 1])

                    img2 = np.concatenate((R, G, B), axis=2)
                    seg_tile_name = os.path.join(seg_dir,base_name)
                    mht.imsave(seg_tile_name,img2)

                    print('Saved {}'.format(seg_tile_name))
                    self.logger.info('Saved {}'.format(seg_tile_name))



    def create_dir_struct(self,root_dir):
        hmap_dir = os.path.join(root_dir,'heat_map')
        if not os.path.exists(hmap_dir):
            os.mkdir(hmap_dir)

        seg_tiles_dir = os.path.join(hmap_dir,'seg_tiles')
        if not os.path.exists(seg_tiles_dir):
            os.mkdir(seg_tiles_dir)



def main():
    if len(sys.argv) != 2:
        print('Usage: TileMasker.py <root_dir>')
        exit()

    root_dir = str(sys.argv[1])  # abs path to where the images are
    tileMasker = TileMasker('Apply tile masks',root_dir)
    tileMasker.apply_masks(root_dir)

if __name__ == '__main__':
    main()
