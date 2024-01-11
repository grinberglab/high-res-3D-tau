from lxml import etree as ET

import sys
import numpy as np
import re


def create_xml(in_file,out_file,imgs_dir):

    folder = ''
    sensorW = ''
    sensorH = ''
    pixel_size = 0
    overlap = 0
    nTilesX = ''
    nTilesY = ''

    nLinesHdr = 8
    PIX_CTS = 0

    with open(out_file,'a'):

        root_xml = ET.Element('TeraStitcher', attrib={'volume_format': 'TiledXY|2Dseries', 'input_plugin': 'tiff2D'})

        with open(in_file) as f:
            lines = f.readlines()
            nLines = len(lines)
            #read header lines
            for l in range(nLinesHdr):
                line = lines[l]
                print(line)
                line = line.strip('\n')

                if 'Folder:' in line:
                    idx = line.find(' ')
                    size = len(line)

                    if imgs_dir == '':
                        folder = line[idx + 1:size]
                    else:
                        folder = imgs_dir #ignores the original folder in Metadata.txt

                    folder_xml = ET.SubElement(root_xml, 'stacks_dir', attrib={'value': folder})

                elif 'Sensor:' in line:
                    sensorW, sensorH = get_nums(line)
                elif 'Pixel:' in line:
                    idx = line.find(' ')
                    size = len(line)
                    pixel_size = line[idx + 1:size]

                    voxel_xml = ET.SubElement(root_xml, 'voxel_dims',
                                              attrib={'V': pixel_size, 'H': pixel_size, 'D': '1'})
                    origin_xml = ET.SubElement(root_xml, 'origin', attrib={'V': '0', 'H': '0', 'D': '0'})

                elif 'Pixel_cts' in line:
                    idx = line.find(' ')
                    size = len(line)
                    pixel_cts = line[idx + 1:size]
                    PIX_CTS = float(pixel_cts)

                elif 'Overlap:' in line:
                    idx = line.find(' ')
                    size = len(line)
                    overlap = line[idx + 1:size]
                    dWidth, dHeight = compute_displacement(float(sensorW), float(sensorH), float(overlap), float(pixel_size))

                    disp_xml = ET.SubElement(root_xml, 'mechanical_displacements',
                                             attrib={'V': str(dHeight), 'H': str(dWidth)})

                elif 'Grid:' in line:
                    nTilesX, nTilesY = get_nums(line)

                    dim_xml = ET.SubElement(root_xml, 'dimensions',
                                            attrib={'stack_rows': nTilesY, 'stack_columns': nTilesX,
                                                    'stack_slices': '1'})

            stacks_xml = ET.SubElement(root_xml, 'STACKS')
            nTilesX = int(nTilesX)
            nTilesY = int(nTilesY)

            #Our scanner's coordinate origin is in the FOV top left corner, while TeraStich's origin is in the bottom left corner
            #we need to read all line into a vector and invert the rows positions, otherwise TeraStitch won't read our XMLs
            #obs: rows MUST appear in the righ order in the XML file otherwise the TereStitch will merge then in incorrect order
            #one must be careful to reverse the rows without reversing the columns, which are in the right order
            #so we read all tiles and create a dictionary with all tiles per row

            #create dictionary with tile info
            # tilesDic = {}
            # currLine = nLinesHdr
            # for r in range(nTilesY):
            #     columns = []
            #     for c in range(nTilesX):
            #         line = lines[currLine]
            #         columns.append(line)
            #         currLine += 1
            #     tilesDic[r] = columns #add to dictionary
            #
            # r = 0
            # for row in range(nTilesY-1,-1,-1): #read dic in reverse order
            #     colLines = tilesDic.get(row)
            #     for c in range(nTilesX):
            #         line = colLines[c]
            #         X,Y = get_nums(line)
            #         X = float(X)
            #         Y = float(Y)
            #         x_vox = int(round(X/PIX_CTS)) #X in voxels
            #         y_vox = int(round(Y/PIX_CTS)) #Y in voxels
            #         fileCount = get_tile_ind(line)
            #         file_name = 'tile_' + str(fileCount) + '.tif'
            #
            #         stack_xml = ET.SubElement(stacks_xml, 'Stack',
            #                                          attrib={'N_CHANS' : '3', 'N_BYTESxCHAN' : '1',
            #                                                 'ROW' : str(r), 'COL' : str(c),
            #                                                 'ABS_V' : str(y_vox), 'ABS_H' : str(x_vox), 'ABS_D' : "0",
            #                                                 'STITCHABLE' : "yes", 'DIR_NAME' : ".",
            #                                                 'Z_RANGES' : '[0,1)',
            #                                                 'IMG_REGEX' : file_name})
            #
            #         ET.SubElement(stack_xml, 'NORTH_displacements')
            #         ET.SubElement(stack_xml, 'EAST_displacements')
            #         ET.SubElement(stack_xml, 'SOUTH_displacements')
            #         ET.SubElement(stack_xml, 'WEST_displacements')
            #
            #     r += 1


            currLine = nLinesHdr
            fileCount = 0
            for r in range(nTilesY):
                for c in range(nTilesX):
                    line = lines[currLine]
                    X,Y = get_nums(line)
                    X = float(X)
                    Y = float(Y)
                    x_vox = int(round(X/PIX_CTS)) #X in voxels
                    y_vox = -1*(int(round(Y/PIX_CTS)))#Y in voxels - I'm multiplying by -1 to reverse the row order
                    file_name = 'tile_' + str(fileCount) + '.tif'

                    stack_xml = ET.SubElement(stacks_xml, 'Stack',
                                             attrib={'N_CHANS' : '3', 'N_BYTESxCHAN' : '1',
                                                    'ROW' : str(r), 'COL' : str(c),
                                                    'ABS_V' : str(y_vox), 'ABS_H' : str(x_vox), 'ABS_D' : "0",
                                                    'STITCHABLE' : "yes", 'DIR_NAME' : ".",
                                                    'Z_RANGES' : '[0,1)',
                                                    'IMG_REGEX' : file_name})

                    north_xml = ET.SubElement(stack_xml, 'NORTH_displacements')
                    east_xml = ET.SubElement(stack_xml, 'EAST_displacements')
                    south_xml = ET.SubElement(stack_xml, 'SOUTH_displacements')
                    west_xml = ET.SubElement(stack_xml, 'WEST_displacements')

                    currLine += 1
                    fileCount += 1


        xml_tree = ET.ElementTree(root_xml)
        print(ET.tostring(xml_tree, pretty_print=True, xml_declaration=True, encoding='UTF-8', doctype='<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">'))
        with open(out_file,'wb') as out:
             out.write(ET.tostring(xml_tree, pretty_print=True, xml_declaration=True, encoding='UTF-8', doctype='<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">'))




def compute_displacement(W,H,overlap,pix_size):
    overlap = overlap / 100  # convert to [0,1]

    # H: width
    over = 2 * W * overlap
    dW_px = W - over #disp in pixels
    dispW = dW_px * pix_size #disp in um

    # V: height
    over2 = 2 * H * overlap
    dH_px = H - over2
    dispH = dH_px * pix_size

    return dispW,dispH

#get numbers between parenthesis
def get_nums(line):
    line = re.sub(r'\s+', '', line)
    idx1 = line.find('(')
    idx2 = line.find(',')
    idx3 = line.find(')')

    num1 = line[idx1+1:idx2]
    num2 = line[idx2+1:idx3]

    return num1,num2

#the the tile index
def get_tile_ind(line):
    line = re.sub(r'\s+', '', line)
    idx1 = line.find(':')
    num1 = line[0:idx1]

    return num1

def main():
    if len(sys.argv) < 3:
        print('Usage: create_terastitch_xml <meta_data_file> <xml_file> <imgs_dir>')
        exit()
    elif len(sys.argv) >= 3:
        in_file = str(sys.argv[1])  # abs path to where the images are
        out_file = str(sys.argv[2])
        imgs_dir = ''
        if len(sys.argv) == 4:
            imgs_dir = str(sys.argv[3])

    print('Input: ' + in_file)
    print('Output: ' + out_file)
    print('Images dir: ' + imgs_dir)

    create_xml(in_file,out_file,imgs_dir)



if __name__ == '__main__':
    main()
