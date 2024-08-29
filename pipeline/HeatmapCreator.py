import os
import fnmatch
import skimage.io as io
from misc.XMLUtils import XMLUtils
import logging
import glob
from misc.TiffTileLoader import TiffTileLoader
import numpy as np
import skimage.transform as xform
import nibabel as nib

TILE_COORDS_FILE = 'tiles/tile_coordinates.npy' #inside output/RES???/, stores tiles coordinates
TILING_INFO_FILE = 'tiles/tiling_info.xml' #inside output/RES???/, stores gridsize and original file size
TILES_ADJ_METADATA = 'heat_map/TAU_seg_tiles/tiles_metadata.xml' # stores tiles adjacency information
TAU_SEG_DIR = 'heat_map/TAU_seg_tiles'
SEG_TILE_DIR = 'heat_map/seg_tiles'
HISTO_TILE_NAME = 'tile_{:04d}.tif'
SEG_TILE_NAME = 'tile_{:04d}_mask.tif'
HMAP_RES = 0.1 # 0.1mm

class HeatmapCreator(object):

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
        log_name = os.path.join(root_dir,'HeatmapCreator.log')
        handler = logging.FileHandler(log_name)
        handler.setLevel(logging.DEBUG)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        self.logger.addHandler(handler)

        #default values
        self.PIX_1MM = 2890  # 1mm= 2890 pixels for 10x RO1
        self.PIX_5MM = 14450  # 5mm = 14450 pixels for 10x RO1
        #self.PIX_1MM = 5814  # 1mm= 5814 pixels for 20x Luke Syn
        #self.PIX_5MM = 29070  # 5mm = 29070 pixels for 20x Luke Syn
        self.HMAP_RES = 0.1
        self.SCALE_FACTOR_VAL = 1000.0
        self.X_DIM = 0.0123
        self.Y_DIM = 0.0123
        self.Z_DIM = 0.122


        self.logger.info('Initializing stage.')


    def get_stage_name(self):
        return self.stage_name

    #run stage method
    def run_stage(self):
        if self.root_dir:

            slice_dirs = self.get_dirs_to_process(self.root_dir)

            self.logger.info('{}  directories to process.'.format(len(slice_dirs)))

            for slice_dir in slice_dirs:
                self.run_compute_heatmap(slice_dir)

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
            self.X_DIM = float(self.config.get('heat_map', 'X_DIM'))
            self.Y_DIM = float(self.config.get('heat_map', 'Y_DIM'))
            self.Z_DIM = float(self.config.get('heat_map', 'Z_DIM'))



    def get_dir_list(self, root_dir):
        list_dirs = glob.glob(root_dir + '/*/')
        return list_dirs


    def get_num_white(self, block):
        # num. non zeros in the blue channel
        tmp_nnz_b = block.flatten().nonzero()
        nnz_b = float(len(tmp_nnz_b[0]))  # number of non-zero pixel in BLOCK matrix
        return nnz_b

    def get_num_pix_tissue(self, img):  # assumes RGB image
        tmp_img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        tmp_nnz_b = tmp_img.flatten().nonzero()
        nnz_b = float(len(tmp_nnz_b[0]))  # number of non-zero pixel in img
        return nnz_b


    #compute heat TAU percentages
    def compute_heatmap(self, TAU_seg_dir, seg_tiles_dir, hm_name, orig_img_size, coords_map):

        #NPIX_BLOCK = pix_mm
        NPIX_BLOCK = self.HMAP_RES*self.PIX_1MM #num os pix in a block at HMAP_RES resolution (along rows or col dimensions)
        NPIX_BLOCK = int(np.round(NPIX_BLOCK))
        min_val_tissue = 0
        max_val_tissue = 0
        self.logger.debug('Number of pixel inside a heatmap block: {}'.format(NPIX_BLOCK))

        #create a memory mapped matriz the size of the original histo image
        #hm_name = os.path.join(hm_dir,'heatmap_{}.npy'.format(slice_id))
        heatmap_per_tissue = np.memmap(hm_name, dtype='float32', mode='w+', shape=(orig_img_size[0],orig_img_size[1]))

        nTiles = coords_map.shape[0]
        self.logger.debug('Number of tiles: {}'.format(nTiles))

        #iterate over each tile
        for nTile in range(nTiles):
            img_name = HISTO_TILE_NAME.format(nTile)
            img_path = os.path.join(seg_tiles_dir, img_name)
            mask_name = SEG_TILE_NAME.format(nTile)
            mask_path = os.path.join(TAU_seg_dir, mask_name)

            #histo_per_tissue = np.zeros(img.shape[0:2])

            #process tile data
            # run image processing routine if TAU tangles mask exists
            if os.path.isfile(mask_path):

                img = io.imread(img_path)

                row_up = coords_map[nTile,0]
                col_up = coords_map[nTile,1]
                row_low = coords_map[nTile,2]
                col_low = coords_map[nTile,3]
                rows_tmp = row_low - row_up
                cols_tmp = col_low - col_up

                mask = io.imread(mask_path)
                rows = mask.shape[0]
                cols = mask.shape[1]

                #sanity check, check if size from coordinates file matches tile image size
                if rows_tmp != rows:
                    self.logger.info('Error: Tile row size differs from size calculated from coordinates. Skipping.')
                    self.nErrors += 1
                    continue
                if cols_tmp != cols:
                    self.logger.info('Error: Tile columns size differs from size caculated from coordinates. Skipping.')
                    self.nErrors += 1
                    continue

                #temporary matrix has the same dimensions of the tile
                histo_per_tissue = np.zeros((rows,cols))

                #compute num. of block in tile along_rows
                nblocks_tile_rows = rows/NPIX_BLOCK
                nblocks_tile_rows = int(np.round(nblocks_tile_rows))

                #compute num. of block along cols
                nblocks_tile_cols = cols/NPIX_BLOCK
                nblocks_tile_cols = int(np.round(nblocks_tile_cols))

                bg_row = 0
                bg_col = 0
                for r in range(nblocks_tile_rows):  # process blocks num. blocks along rows
                    end_row = NPIX_BLOCK * (r + 1)

                    for c in range(nblocks_tile_cols): # num. blocks along cols
                        end_col = NPIX_BLOCK * (c + 1)

                        # last block can be problematic, we have to check if the indices are inside the right range
                        if c == (nblocks_tile_cols - 1):
                            if end_col != cols:
                                end_col = cols

                        if r == (nblocks_tile_rows - 1):
                            if end_row != rows:
                                end_row = rows

                        block_mask = mask[bg_row:end_row, bg_col:end_col]
                        block_img = img[bg_row:end_row, bg_col:end_col, :]

                        nonzero_pix_mask = self.get_num_white(block_mask)  # get number of non-zero pixels in mask
                        #total_pix_block = rows * cols  # total number of pixel in image block
                        npix_tissue_block = self.get_num_pix_tissue(block_img)

                        #percent_total = float(nonzero_pix_mask) / float(total_pix_block)
                        percent_tissue = 0.0 if npix_tissue_block == 0 else (
                                float(nonzero_pix_mask) / float(npix_tissue_block))

                        histo_per_tissue[bg_row:end_row, bg_col:end_col] = percent_tissue * 100

                        bg_col = end_col

                    bg_row = end_row
                    bg_col = 0

                # get min and max values for the entire slice, per amount of tissue
                if histo_per_tissue.min() < min_val_tissue:
                    min_val_tissue = histo_per_tissue.min()
                if histo_per_tissue.max() > max_val_tissue:
                    max_val_tissue = histo_per_tissue.max()

                # hm2_name = os.path.join(hm_dir, img_name[0:-4] + '_hm_pertissue.npy')
                # np.save(hm2_name, histo_per_tissue)
                heatmap_per_tissue[row_up:row_low,col_up:col_low] = histo_per_tissue[...]

        self.logger.info('Saving full resolution heatmap.')
        heatmap_per_tissue.flush()
        del heatmap_per_tissue

        self.logger.debug('Max val:{} Min val:{}'.format(max_val_tissue,min_val_tissue))
        return min_val_tissue, max_val_tissue


    #rescale heatmap
    def rescale_heatmap(self, res10_file, hm_file, res10_hm_file, full_coords_map, grid_tile, orig_img_size,SCALE_FACTOR = 1.0):
        tiffLoader_res10 = TiffTileLoader()
        tiffLoader_res10.open_file(res10_file)
        tiffLoader_res10.compute_tile_coords(grid_tile[0],grid_tile[1])
        res10_coords_map = tiffLoader_res10.get_tile_coords()

        #sanity check
        nTiles_s = res10_coords_map.shape[0] #num tiles computed for res10 image
        nTiles_f = full_coords_map.shape[0] #num tiles computed for full res image
        if nTiles_f != nTiles_s:
            self.logger.info('Number of tiles from full res images and res10 images does not match.')
            self.nErrors += 1
            return False

        #get res10 image size
        res10_size = tiffLoader_res10.get_file_dim()

        #open heatmap full res file
        hmap_full = np.memmap(hm_file, dtype='float32', mode='r', shape=(orig_img_size[0],orig_img_size[1]))

        #create res10 heatmap file
        #hmap_res10 = np.memmap(res10_hm_file, dtype='float32', mode='w+', shape=(res10_size[0],res10_size[1]))
        hmap_res10 = np.zeros((res10_size[0],res10_size[1]))

        #computed res10 heatmap
        for nT in range(nTiles_s):
            row_up_full, col_up_full, row_low_full, col_low_full = full_coords_map[nT,...]
            row_up_res10, col_up_res10, row_low_res10, col_low_res10 = res10_coords_map[nT,...]

            new_row_size = row_low_res10 - row_up_res10
            new_col_size = col_low_res10 - col_up_res10

            full_tile = hmap_full[row_up_full:row_low_full,col_up_full:col_low_full].copy() #be carefull, hmap_full is read-only
            full_tile *= SCALE_FACTOR
            res10_tile = xform.resize(full_tile,(new_row_size,new_col_size),order=0,clip=False,preserve_range=True)
            hmap_res10[row_up_res10:row_low_res10,col_up_res10:col_low_res10] = res10_tile[...]

        self.logger.info('Saving res 10 numpy heatmap file.')
        np.save(res10_hm_file,hmap_res10)
        #hmap_res10.flush()
        del hmap_full
        #del hmap_res10

        #create NIFTI heatmap file
        self.logger.info('Saving res 10 NIFTI file.')
        nii_name = res10_hm_file[0:-4] + '.nii'
        M = np.eye(4) * np.array([self.X_DIM, self.Y_DIM, self.Z_DIM, 1])
        nii = nib.Nifti1Image(hmap_res10,M)
        nib.save(nii,nii_name)

        #save scale factor
        self.logger.info('Saving scale factor file (Scale factor: {}).'.format(SCALE_FACTOR))
        scale_name = res10_hm_file[0:-4] + '_scale.npy'
        sfactor = np.array([SCALE_FACTOR])
        np.save(scale_name,sfactor)
        return True


    #get information from XML file created during the histology tiling stage (ImageTiler.py)
    def load_tiling_metadata(self, meta_xml):
        grid_rows, grid_cols, img_rows, img_cols, img_home, img_file = XMLUtils.parse_tiles_metadata(meta_xml)
        return [grid_rows,grid_cols],[img_rows,img_cols]


    #get tile coordinates from the file created during the histology tiling processes (ImageTiler.py)
    def load_tile_coords(self,coords_file):
        coords = np.load(coords_file).astype('int')
        return coords


    def get_res_dir(self,root_dir):
        res_dir = ''
        res10_file = ''
        for root, dir, files in os.walk(root_dir):
            if fnmatch.fnmatch(root,'*/RES(*'): #it's inside /RES*
                if res_dir == '':
                    res_dir = root
                for fn in fnmatch.filter(files,'*.tif'): #get only full resolution images
                    if fn.find('res10') > -1: #skip res10 images
                        res10_file = fn

        res10_file = os.path.join(res_dir,res10_file)
        return res_dir,res10_file

    def get_dirs_to_process(self,root_dir):
        dirs_list = []
        slice_dirs = glob.glob(os.path.join(root_dir, '*'))
        for sd in slice_dirs:
            if (os.path.isdir(sd) or os.path.islink(sd)) and sd.find('magick_tmp') == -1:
                dirs_list.append(sd)
        return dirs_list


    def run_compute_heatmap(self,slice_dir):
        self.logger.info('*** Beginning to compute heatmap {} ***'.format(slice_dir)) #i.e. /.../.../batch1/AT100_456/
        print('*** Beginning to compute heatmap {} ***'.format(slice_dir))  # i.e. /.../.../batch1/AT100_456/
        #path = os.path.normpath(slice_dir).split(os.sep)
        #slice_id = path[-1]
        tau_seg_dir = os.path.join(slice_dir,TAU_SEG_DIR)
        seg_tiles_dir = os.path.join(slice_dir,SEG_TILE_DIR)
        hm_dir = os.path.join(slice_dir,'heat_map/hm_map_'+str(self.HMAP_RES))

        if not os.path.exists(hm_dir):
            os.mkdir(hm_dir)
        hm_file = os.path.join(hm_dir,'heat_map_'+str(self.HMAP_RES)+'.npy')

        res_dir, res10_file = self.get_res_dir(slice_dir)
        xml_file = os.path.join(res_dir,TILING_INFO_FILE)
        coords_file = os.path.join(res_dir,TILE_COORDS_FILE)

        self.logger.debug('TAU_seg_dir: {}'.format(tau_seg_dir))
        self.logger.debug('seg_tiles_dir: {}'.format(seg_tiles_dir))
        self.logger.debug('heatmap_dir: {}'.format(hm_dir))
        self.logger.debug('RES(?x?)_dir: {}'.format(res_dir))
        self.logger.debug('res10 image files: {}'.format(res10_file))
        self.logger.debug('Full-res coords file: {}'.format(coords_file))


        #load coords map
        self.logger.info('Loading coordinates map.')
        print('Loading coordinates map.')
        coords_map = self.load_tile_coords(coords_file)
        self.logger.info('Coordinate maps successfully loaded.')

        #load tiling metadata
        self.logger.info('Loading tiling metadata.')
        print('Loading tiling metadata.')
        grid_tile,orig_img_size = self.load_tiling_metadata(xml_file)
        self.logger.info('Tiling metadata successfully loaded.')

        #run heatmap computation
        self.logger.info('Beginning to compute heatmap.')
        print('Beginning to compute heatmap.')
        min_val_tissue, max_val_tissue = self.compute_heatmap(tau_seg_dir, seg_tiles_dir, hm_file, orig_img_size, coords_map)
        self.logger.info('Heatmap computation finished.')

        #save min-max file
        self.logger.info('Creating min/max file (min: {}, max:{}).'.format(min_val_tissue, max_val_tissue))
        print('Creating min/max file (min: {}, max:{}).'.format(min_val_tissue, max_val_tissue))
        min_max_file = os.path.join(hm_dir,'min_max.npy')
        min_max_arr = np.array([min_val_tissue, max_val_tissue])
        np.save(min_max_file, min_max_arr)
        self.logger.info('Min/max file successfully saved.')

        #compute res10 heatmap
        self.logger.info('Creating 10% resolution heatmap.')
        print('Creating 10% resolution heatmap.')
        res10_hm_file = os.path.join(hm_dir,'heat_map_'+str(self.HMAP_RES)+'_res10.npy')
        status = self.rescale_heatmap(res10_file, hm_file, res10_hm_file, coords_map, grid_tile, orig_img_size, self.SCALE_FACTOR_VAL)
        if status:
            self.logger.info('10% resolution heatmap successfully created.')
        else:
            self.logger.info('There was ans error creating 10% resolution heatmap.')



