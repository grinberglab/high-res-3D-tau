import skimage.feature as ft
import skimage.io as io
from sklearn.feature_extraction import image
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import image as skimg

# from http://jamesgregson.ca/extract-image-patches-in-python.html

def extract_grayscale_patches( img, shape, offset=(0,0), stride=(1,1) ):
    """Extracts (typically) overlapping regular patches from a grayscale image

    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!

    Args:
        img (HxW ndarray): input image from which to extract patches

        shape (2-element arraylike): shape of that patches as (h,w)

        offset (2-element arraylike): offset of the initial point as (y,x)

        stride (2-element arraylike): vertical and horizontal strides

    Returns:
        patches (ndarray): output image patches as (N,shape[0],shape[1]) array

        origin (2-tuple): array of top and array of left coordinates
    """
    px, py = np.meshgrid( np.arange(shape[1]),np.arange(shape[0]))
    l, t = np.meshgrid(
        np.arange(offset[1],img.shape[1]-shape[1]+1,stride[1]),
        np.arange(offset[0],img.shape[0]-shape[0]+1,stride[0]) )
    l = l.ravel()
    t = t.ravel()
    x = np.tile( px[None,:,:], (t.size,1,1)) + np.tile( l[:,None,None], (1,shape[0],shape[1]))
    y = np.tile( py[None,:,:], (t.size,1,1)) + np.tile( t[:,None,None], (1,shape[0],shape[1]))
    return img[y.ravel(),x.ravel()].reshape((t.size,shape[0],shape[1])), (t,l)

def reconstruct_from_grayscale_patches( patches, origin, epsilon=1e-12 ):
    """Rebuild an image from a set of patches by averaging

    The reconstructed image will have different dimensions than the
    original image if the strides and offsets of the patches were changed
    from the defaults!

    Args:
        patches (ndarray): input patches as (N,patch_height,patch_width) array

        origin (2-tuple): top and left coordinates of each patch

        epsilon (scalar): regularization term for averaging when patches
            some image pixels are not covered by any patch

    Returns:
        image (ndarray): output image reconstructed from patches of
            size ( max(origin[0])+patches.shape[1], max(origin[1])+patches.shape[2])

        weight (ndarray): output weight matrix consisting of the count
            of patches covering each pixel
    """
    patch_width  = patches.shape[2]
    patch_height = patches.shape[1]
    img_width    = np.max( origin[1] ) + patch_width
    img_height   = np.max( origin[0] ) + patch_height

    out = np.zeros( (img_height,img_width) )
    wgt = np.zeros( (img_height,img_width) )
    for i in range(patch_height):
        for j in range(patch_width):
            out[origin[0]+i,origin[1]+j] += patches[:,i,j]
            wgt[origin[0]+i,origin[1]+j] += 1.0

    return out/np.maximum( wgt, epsilon ), wgt



def main():
    test_img = '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/debug_training/images/patch_0_242297.tif'
    img = io.imread(test_img)
    img_gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    p, origin = extract_grayscale_patches( img_gry, (8,8), stride=(1,1) )

    tex_corr = np.zeros((p.shape[0],p.shape[1],p.shape[2]))
    tex_diss = np.zeros((p.shape[0], p.shape[1], p.shape[2]))
    tex_cont = np.zeros((p.shape[0], p.shape[1], p.shape[2]))

    nPatches = len(p)
    for i in range(nPatches):
        pat = p[i,...]
        glcm = ft.greycomatrix(pat,[5], [0], 256, symmetric=True, normed=True)
        diss = ft.greycoprops(glcm, 'dissimilarity')[0, 0]
        corr = ft.greycoprops(glcm, 'correlation')[0, 0]
        cont = ft.greycoprops(glcm, 'contrast')[0, 0]
        tex_corr[i,...] = corr
        tex_diss[i,...] = diss
        tex_cont[i,...] = cont


    rcorr, w1 = reconstruct_from_grayscale_patches(tex_corr, origin )
    rdiss, w2 = reconstruct_from_grayscale_patches(tex_diss, origin)
    rcont, w3 = reconstruct_from_grayscale_patches(tex_cont, origin)

    plt.imshow(rcorr)



    # plt.subplot( 131 )
    # plt.imshow( img[:,:,2] )
    # plt.title('Input image')
    # plt.subplot( 132 )
    # plt.imshow( p[p.shape[0]//2] )
    # plt.title('Central patch')
    # plt.subplot( 133 )
    # plt.imshow( r )
    # plt.title('Reconstructed image')
    # plt.show()



if __name__ == '__main__':
    main()