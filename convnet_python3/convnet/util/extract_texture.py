import glob
import os
import numpy as np
import argparse
import skimage.io as io
import skimage.feature as ft
from convnet.util.help_functions import  extract_grayscale_patches, reconstruct_from_grayscale_patches
import cv2


def lin_scale(arr,min_max,range):
    arr2 = (((range[1] - range[0])*(arr - min_max[0]))/(min_max[1] - min_max[0])) + range[0]
    return arr2


def compute_textures(tif_dir):
    list_files = glob.glob(os.path.join(tif_dir,'*tif'))

    for f in list_files:
        img = io.imread(f)
        img_gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        p, origin = extract_grayscale_patches(img_gry, (8, 8), stride=(1, 1))
        nTex = 6

        tex_tmp = np.zeros((p.shape[0],p.shape[1],p.shape[2],nTex))
        tex_maps = np.zeros((img.shape[0],img.shape[1],nTex))
        min_max = -1*np.ones((2,nTex))

        nPatches = len(p)

        #compute texture
        for i in range(nPatches):
            pat = p[i, ...]
            glcm = ft.greycomatrix(pat, [5], [0], 256, symmetric=True, normed=True)

            cont = ft.greycoprops(glcm, 'contrast')[0, 0]
            diss = ft.greycoprops(glcm, 'dissimilarity')[0, 0]
            hom = ft.greycoprops(glcm, 'homogeneity')[0, 0]
            corr = ft.greycoprops(glcm, 'correlation')[0, 0]
            asm = ft.greycoprops(glcm, 'ASM')[0, 0]
            ene = ft.greycoprops(glcm, 'energy')[0, 0]

            tex_tmp[i,:,:,0] = cont
            tex_tmp[i, :, :, 1] = diss
            tex_tmp[i, :, :, 2] = hom
            tex_tmp[i, :, :, 3] = corr
            tex_tmp[i, :, :, 4] = asm
            tex_tmp[i, :, :, 5] = ene

        #reshape and compute [min max]
        for i in range(nTex):
            tex_maps[:, :, i], w = reconstruct_from_grayscale_patches(tex_tmp[..., 0], origin)
            min_max[0,i] = tex_maps[...,i].min()
            min_max[1,i] = tex_maps[...,i].max()

        #rescale to [-1,1]
        for i in range(nTex):
            tex_maps[...,i] = lin_scale(tex_maps[...,i], min_max[:,i], [-1,1])
            

        basename = os.path.basename(f)
        dirname = os.path.dirname(basename)
        tex_basename = basename[:-4]+'_tex.npy'
        tex_name = os.path.join(dirname,tex_basename)
        np.save(tex_name,tex_maps)

        minmax_name = os.path.join(dirname,'min_max_texture.npy')
        np.save(minmax_name)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d','--dir', required=True, help='directory with tif files')
    args = vars(ap.parse_args())
    tif_dir = args['dir']







if __name__ == '__main__':
    main()