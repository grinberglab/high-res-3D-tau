from lxml import etree as ET
import fnmatch
import os
from skimage import io
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import tifffile


PIX_MM = 819




def sub2ind(size,r,c):
    ind = r*size[1]+c
    return ind

def create_adj_dic(grid_shape):
    rows = grid_shape[0]
    cols = grid_shape[1]

    tiles_dic = {}
    for r in range(rows):
        for c in range(cols):

            N = -1
            S = -1
            E = -1
            W = -1
            curr = sub2ind(grid_shape,r,c)
            #get north
            if r > 0:
                N = sub2ind(grid_shape,r-1,c)
            #get south
            if r < rows:
                S = sub2ind(grid_shape,r+1,c)
            #get west
            if c > 0:
                W = sub2ind(grid_shape,r,c-1)
            #get east
            if c < cols:
                E = sub2ind(grid_shape,r,c+1)

            tiles_dic[curr] = np.array([N,S,E,W])

    return tiles_dic


def create_xml_metadata(tile_dir, seg_dir, grid_rows, grid_cols, img_rows, img_cols, nblocks_tile,
                        file_pt = 'tile_{:04d}.tif', file_pt_mask = 'tile_{:04d}_mask.tif', pix_mm=PIX_MM):
    grid_shape = np.array([grid_rows,grid_cols])
    tiles_dic = create_adj_dic(grid_shape)

    root_xml = ET.Element('Tiles', attrib={'grid_rows': str(grid_rows), 'grid_cols': str(grid_cols),
                                           'img_rows': str(img_rows), 'img_cols': str(img_cols),
                                           'pix_mm': str(pix_mm), 'nblocks_tile': str(nblocks_tile),
                                           'root_dir': tile_dir})
    keys = tiles_dic.keys()
    for tile_num in keys:
        #load current image
        img_name = file_pt.format(tile_num) #tile
        img_path = os.path.join(tile_dir, img_name)
        mask_name = file_pt_mask.format(tile_num) #segmented tile
        mask_path = os.path.join(seg_dir,mask_name)

        try:
            img = tifffile.TiffFile(mask_path) #mask exists
            is_mask = 1
            img_name = mask_name
        except:
            try:
                img = tifffile.TiffFile(img_path)
                is_mask = 0
            except:
                print('Warning: Tile {} not found. Skipping it.'.format(tile_num))
                continue

        img_shape = img.series[0].shape
        tile_xml = ET.SubElement(root_xml, 'Tile', attrib={'is_mask':str(is_mask), 'name':img_name,
                                                           'rows':str(img_shape[0]), 'cols':str(img_shape[1])})

        nbors = tiles_dic[tile_num] #Neighbors are always in N,S,E,W order
        #get north:
        if nbors[0] != -1:
            n_name = file_pt.format(nbors[0])
            N_tile_xml = ET.SubElement(tile_xml, 'North', attrib={'name':n_name})

        #get south:
        if nbors[1] != -1:
            s_name = file_pt.format(nbors[1])
            S_tile_xml = ET.SubElement(tile_xml, 'South', attrib={'name':s_name})

        #get east:
        if nbors[2] != -1:
            e_name = file_pt.format(nbors[2])
            E_tile_xml = ET.SubElement(tile_xml, 'East', attrib={'name':e_name})

        #get west:
        if nbors[3] != -1:
            w_name = file_pt.format(nbors[3])
            W_tile_xml = ET.SubElement(tile_xml, 'West', attrib={'name':w_name})



    xml_tree = ET.ElementTree(root_xml)
    return xml_tree

def find_xml_file(case_dir):
    xml_file = ''
    for root, dir, files in os.walk(case_dir):
        if fnmatch.fnmatch(root,'*/RES(*'): #it's inside /RES*
            tmp_path = os.path.join(root,'tiles/tiling_info.xml')
            if os.path.exists(tmp_path):
                xml_file = tmp_path
                break

    return xml_file

def get_info_xml(xml_file):
    xml_tree = ET.parse(xml_file)

    image_node = xml_tree.xpath('//Image')[0]
    img_rows = int(image_node.attrib['rows'])
    img_cols = int(image_node.attrib['cols'])
    tiles_node = xml_tree.xpath('//Tiles')[0]
    # orig_rows = int(tiles_node.attrib['rows'])
    # orig_cols = int(tiles_node.attrib['cols'])
    #pix_mm = int(tiles_node.attrib['pix_mm'])
    pix_mm = PIX_MM
    #nblocks = int(tiles_node.attrib['nblocks_tile'])
    nblocks = 5
    grid_cols = int(tiles_node.attrib['grid_cols'])
    grid_rows = int(tiles_node.attrib['grid_rows'])

    return grid_rows,grid_cols,nblocks,pix_mm,img_rows,img_cols


def export_metadata(case_dir):
    xml_file = os.path.join(case_dir, 'heat_map/TAU_seg_tiles/tiles_metadata.xml')
    tiles_dir = os.path.join(case_dir, 'heat_map/seg_tiles/')
    seg_dir = os.path.join(case_dir, 'heat_map/TAU_seg_tiles/')

    if not os.path.exists(seg_dir):
        print('{} does not exist.'.format(seg_dir))
        return False

    # get info from tiling metadata
    tiling_meta_xml = find_xml_file(case_dir)
    grid_rows, grid_cols, nblocks, pix_mm, img_rows, img_cols = get_info_xml(tiling_meta_xml)

    print('Exporting tiles metadata...')
    xml_tree = create_xml_metadata(tiles_dir, seg_dir, grid_rows, grid_cols, img_rows, img_cols, nblocks)
    print(ET.tostring(xml_tree, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    with open(xml_file, 'w+') as out:
        out.write(ET.tostring(xml_tree, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    print('Metadata saved in {}'.format(xml_file))

    return True




def main():
    #The program will extract the information from the tiling_info.xml that should be inside
    #<CASE_DIR>/output/RES(???x???)/tiles, if file doesn't exist it will quit
    if len(sys.argv) != 2:
        print('Usage: export_heatmap_metadata.py <case_dir>')
        exit()
    #case_dir='/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res/resize_mary/AT100#648'
    case_dir = str(sys.argv[1])  # abs path to where the images are
    export_metadata(case_dir)
    # xml_file = os.path.join(case_dir,'heat_map/TAU_seg_tiles/tiles_metadata.xml')
    # tiles_dir = os.path.join(case_dir,'heat_map/TAU_seg_tiles/')
    #
    # #get info from tiling metadata
    # tiling_meta_xml = find_xml_file(case_dir)
    # grid_rows, grid_cols, nblocks, pix_mm, img_rows, img_cols = get_info_xml(tiling_meta_xml)
    #
    # print('Exporting tiles metadata...')
    # #tile_dir = '/home/maryana/storage/Posdoc/AVID/AV13/AT100440/TAU_seg_tiles'
    # #xml_file = '/home/maryana/storage/Posdoc/AVID/AV13/AT100440/TAU_seg_tiles/tiles_metadata.xml'
    # #nblocks = 5
    # #cols = 20
    # #rows = 9
    #
    # xml_tree = create_xml_metadata(tiles_dir, grid_rows, grid_cols, img_rows, img_cols, nblocks)
    # print(ET.tostring(xml_tree, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    # with open(xml_file, 'w+') as out:
    #     out.write(ET.tostring(xml_tree, pretty_print=True, xml_declaration=True, encoding='UTF-8'))
    # print('Metadata saved in {}'.format(xml_file))

if __name__ == '__main__':
    main()