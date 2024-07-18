import os
import sys
import skimage.io as io
import tifffile
import glob
from osgeo import gdal
import logging
import numpy as np


def ind2sub(array_shape, ind):
    rows = (int(ind) / array_shape[1])
    cols = (int(ind) % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def sub2ind(size,r,c):
    ind = r*size[1]+c
    return ind


class TiffTileLoader(object):

    def __init__(self, p1MM=5814, p5MM=29070):

        self.ds = None
        gdal.UseExceptions()
        #default values
        # self.PIX_1MM = 819  # 1mm= 819 pixels
        # self.PIX_5MM = 4095  # 5mm = 4095 pixels
        self.PIX_1MM = p1MM
        self.PIX_5MM = p5MM
        self.coords = []
        self.nChannels = -1


    def open_file(self,file_name):
        self.ds = gdal.Open(file_name)
        self.nChannels = self.ds.RasterCount
        if self.nChannels == 3:
            self.rCh = self.ds.GetRasterBand(1)
            self.gCh = self.ds.GetRasterBand(2)
            self.bCh = self.ds.GetRasterBand(3)
            self.bw = None
        else:
            self.bw = self.ds.GetRasterBand(1)
            self.rCh = None
            self.gCh = None
            self.bCh = None
            self.nChannels = 1


    def get_file_dim(self):
        cols = self.ds.RasterXSize
        rows = self.ds.RasterYSize
        z = self.ds.RasterCount

        return [rows,cols,z]


    def get_tile(self, x, y, xsize, ysize):
        x = int(x)
        y = int(y)
        xsize = int(xsize)
        ysize = int(ysize)

        if self.nChannels > 1:
            R = self.rCh.ReadAsArray(x, y, xsize, ysize)
            G = self.gCh.ReadAsArray(x, y, xsize, ysize)
            B = self.bCh.ReadAsArray(x, y, xsize, ysize)
            s = R.shape
            img = np.concatenate((R.reshape(s[0],s[1],1),G.reshape(s[0],s[1],1),B.reshape(s[0],s[1],1)),axis=2)
        else:
            img = self.bw.ReadAsArray(x, y, xsize,ysize)

        return img

    def get_tile_by_num(self,ind):
        if len(self.coords) > 0:
            if ind < 0 or ind > self.coords.shape[0]-1:
                print('Error: index must be a number between 0 and {}'.format(self.coords.shape[0]-1))
                return None

            r1,c1,r2,c2 = self.coords[ind]
            ysize = r2-r1
            xsize = c2-c1
            tile = self.get_tile(c1,r1,xsize,ysize)
            return tile
        else:
            print('Error: Tile coordinates not set.')
            return None


    def compute_tile_coords(self,grid_rows,grid_cols):

        grid_rows = float(grid_rows)
        grid_cols = float(grid_cols)

        size = self.get_file_dim()

        #compute row coords
        tile_coords = np.zeros([(int(grid_rows)*int(grid_cols)),4]) #[row_upper_left, col_upper_left, row_lower_right, col_lower_right]

        row_off = np.floor(size[0]/grid_rows) #initial block size
        #self.logger.debug(print('row_off = {}.'.format(row_off)))
        row_off = int(row_off)
        row_rem = size[0] % int(row_off)
        #self.logger.debug(print('row_rem = {}.'.format(row_rem)))
        row_add = np.zeros([int(grid_rows),1]) #correction factor

        if row_rem > 0: # we have to compensate for uneven block sizes (reminder > 0). Make the last block in the row bigger since it's more likely to be background.
            row_add[-1] = row_rem
            #self.logger.debug(print('row_add = {}.'.format(row_add)))

        col_off = np.floor(size[1]/grid_cols) #initial block size
        #self.logger.debug(print('col_off = {}.'.format(col_off)))
        col_off = int(col_off)
        col_rem = size[1] % int(col_off)
        #self.logger.debug(print('col_rem = {}.'.format(col_rem)))
        col_add = np.zeros([int(grid_cols),1]) #correction factor
        if col_rem > 0:
            col_add[-1] = col_rem
            #self.logger.debug(print('col_add = {}.'.format(col_add)))

        tile_ind = 0
        up_row = 0
        up_col = 0
        for row_count in range(int(grid_rows)):
            for col_count in range(int(grid_cols)):
                #left upper corner
                tile_coords[tile_ind,0] = up_row
                tile_coords[tile_ind,1] = up_col

                low_row = up_row + row_off + row_add[row_count]
                low_col = up_col + col_off + col_add[col_count]
                tile_coords[tile_ind,2] = low_row
                tile_coords[tile_ind,3] = low_col

                #up_col = ((col_count+1) * col_off) + np.sum(col_add[0:col_count + 1])
                up_col = low_col
                tile_ind += 1
            up_col = 0
            #up_row = ((row_count+1) * row_off) + np.sum(row_add[0:row_count + 1])
            up_row = low_row

        self.coords = tile_coords.astype('int')
        #return tile_coords


    def merge_tiles_rgb(self,tiles_dir,coords_file,out_file,grid_tiles,orig_size):
        if not self._sanity_check(tiles_dir,grid_tiles[0],grid_tiles[1],orig_size):
            print('Cannot merge tiles. Tiles size differs from original size.')
            return False

        tile_coords = np.load(coords_file)
        if not self._coords_sanity_check(tile_coords,grid_tiles[0],grid_tiles[1],orig_size):
            print('Cannot merge tiles. Coordinates in coordinates files yields erroneous image size.')
            return False

        img_map = tifffile.memmap(out_file,shape=(orig_size[0],orig_size[1],3),dtype='uint8') #RGB
        files = glob.glob(os.path.join(tiles_dir,'*.tif'))
        for fi in files:
            img = io.imread(fi)
            basename = os.path.basename(fi)
            idx1 = basename.find('_')
            idx2 = basename.find('.tif')
            num = basename[idx1+1:idx2]

            ind = int(num)
            up_row,up_col,low_row,low_col = tile_coords[ind,:].astype('int')
            img_map[up_row:low_row,up_col:low_col,:] = img[...]

        img_map.flush()
        return True


    def get_tile_coords(self):
        return self.coords


    def get_tile_iterator(self):
        iterator = TileIterator(self.ds,self.coords)
        return iterator


    def coords_sanity_check(self,grid_rows,grid_cols):
        size = self.get_file_dim()
        return self._coords_sanity_check(self.coords,grid_rows,grid_cols,(size[0],size[1]))


    def _coords_sanity_check(self,coords_arr, grid_rows,grid_cols, orig_size):
        ok = True

        rows_mat = np.zeros([int(grid_rows),int(grid_cols)]) #tiles height
        cols_mat = np.zeros([int(grid_rows),int(grid_cols)]) #tiles width
        #orig_size = self.get_file_dim()

        for row in range(int(grid_rows)):
            for col in range(int(grid_cols)):
                ind = sub2ind((int(grid_rows),int(grid_cols)),row,col)
                up_row, up_col, low_row, low_col = coords_arr[ind,:] #[up_row, up_col, low_row, low_col]
                cols_mat[row,col] = low_col - up_col #tile width
                rows_mat[row,col] = low_row - up_row #tile height

        for rr in range(int(grid_rows)): #check each row's width
            width = np.sum(cols_mat[rr,:])
            if width != orig_size[1]: #num. cols == width
                print('Coord: Row {} width ({}) differs from original image size ({}).'.format(rr, width, orig_size[1]))
                ok = False

        for cc in range(int(grid_cols)): #check each row's width
            height = np.sum(rows_mat[:,cc])
            if height != orig_size[0]: #num. rows == height
                print('Coord: Column {} height ({}) differs from original image size ({}).'.format(cc, height, orig_size[0]))
                ok = False

        return ok


    def sanity_check(self,tiles_dir,grid_rows,grid_cols):
        orig_size = self.get_file_dim()
        return self._sanity_check(tiles_dir,grid_rows,grid_cols,orig_size)


    def _sanity_check(self,tiles_dir,grid_rows,grid_cols,orig_size):
        ok = True

        rows_mat = np.zeros([int(grid_rows),int(grid_cols)]) #tiles height
        cols_mat = np.zeros([int(grid_rows),int(grid_cols)]) #tiles width
        #orig_size = self.get_file_dim()

        for row in range(int(grid_rows)):
            for col in range(int(grid_cols)):
                ind = sub2ind((int(grid_rows),int(grid_cols)),row,col)
                f = os.path.join(tiles_dir,'tile_{:04}.tif'.format(ind))
                tiff = tifffile.TiffFile(f)  # load tiff header only
                size = tiff.series[0].shape

                rows_mat[row,col] = size[0]
                cols_mat[row,col] = size[1]

        for rr in range(int(grid_rows)): #check each row's width
            width = np.sum(cols_mat[rr,:])
            if width != orig_size[1]: #num. cols == width
                print('Row {} width ({}) differs from original image size ({}).'.format(rr, width, orig_size[1]))
                ok = False

        for cc in range(int(grid_cols)): #check each row's width
            height = np.sum(rows_mat[:,cc])
            if height != orig_size[0]: #num. rows == height
                print('Column {} height ({}) differs from original image size ({}).'.format(cc, height, orig_size[0]))
                ok = False

        return ok



class TileIterator(object):

    def __init__(self,ds,tile_coords):
        self.ds = ds
        self.tile_coords = tile_coords
        self.nTiles = tile_coords.shape[0]
        self.curr = 0

        self.nChannels = self.ds.RasterCount

        if self.nChannels == 3:
            self.rCh = self.ds.GetRasterBand(1)
            self.gCh = self.ds.GetRasterBand(2)
            self.bCh = self.ds.GetRasterBand(3)
            self.bw = None
        else:
            self.bw = self.ds.GetRasterBand(1)
            self.rCh = None
            self.gCh = None
            self.bCh = None
            self.nChannels = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr >= self.nTiles:
            raise StopIteration
        else:
            r1,c1,r2,c2 = self.tile_coords[self.curr]
            ysize = r2-r1
            xsize = c2-c1
            tile = self.get_tile(c1,r1,xsize,ysize)
            self.curr += 1
            return tile

    def get_tile(self, x, y, xsize, ysize):

        x = int(x)
        y = int(y)
        xsize = int(xsize)
        ysize = int(ysize)

        if self.nChannels > 1:
            R = self.rCh.ReadAsArray(x, y, xsize,ysize)
            G = self.gCh.ReadAsArray(x, y, xsize, ysize)
            B = self.bCh.ReadAsArray(x, y, xsize, ysize)
            s = R.shape
            img = np.concatenate((R.reshape(s[0],s[1],1),G.reshape(s[0],s[1],1),B.reshape(s[0],s[1],1)),axis=2)
        else:
            img = self.bw.ReadAsArray(x, y, xsize,ysize)

        return img











