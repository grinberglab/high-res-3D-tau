#Python
from keras.models import model_from_json
from convnet.util.help_functions import *
from convnet.util.extract_patches import get_data_testing_overlap,get_data_segmenting_overlap
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
from vis.utils import utils
from vis.visualization import visualize_cam
from keras.activations import linear
import tensorflow as tf
from tensorflow.python.framework import ops
from misc.TiffTileLoader import sub2ind,ind2sub
from misc.imoverlay import  imoverlay
from mahotas import bwperim
import matplotlib as mpl
import matplotlib.cm as cm
from skimage import img_as_ubyte
import ConfigParser as cparser
import matplotlib as mpl
import matplotlib.cm as cm
import os

from misc.imoverlay import imoverlay as imoverlay
import mahotas as mh
from mahotas import bwperim

class GradCAM:

    def __init__(self,config_file):
        # read config
        config = cparser.RawConfigParser()
        config.read(config_file)
        path_project = config.get('data paths', 'path_project')
        path_model = os.path.join(path_project, config.get('data paths', 'path_model'))

        # dimension of the patches
        self.patch_height = int(config.get('data attributes', 'patch_height'))
        self.patch_width = int(config.get('data attributes', 'patch_width'))
        self.mask_height = int(config.get('data attributes', 'mask_height'))
        self.mask_width = int(config.get('data attributes', 'mask_width'))
        self.mask_dim = (self.mask_height, self.mask_width)
        self.imgs_to_test = int(config.get('testing settings', 'full_images_to_test')),

        # model name
        self.name_experiment = config.get('experiment name', 'name')
        self.average_mode = config.getboolean('testing settings', 'average_mode')
        self.stride_height = int(config.get('testing settings', 'stride_height'))
        self.stride_width = int(config.get('testing settings', 'stride_width'))

        # load mean image for pre-processing
        self.mean_img_path = os.path.join(path_project, config.get('data paths', 'mean_image'))

        # Load the saved model
        self.model = model_from_json(open(os.path.join(path_model, self.name_experiment + '_architecture.json')).read())
        self.model.load_weights(os.path.join(path_model, self.name_experiment + '_best_weights.h5'))


    def ind2sub(self, array_shape, ind):
        rows = (int(ind) / array_shape[1])
        cols = (int(ind) % array_shape[1])
        return (rows, cols)

    def sub2ind(self, size, r, c):
        ind = r * size[1] + c
        return ind

    def deprocess_image(self,x):
        """From:
        https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
        """
        x = x.copy()
        if np.ndim(x) > 3:
            x = np.squeeze(x)
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        # if K.image_dim_ordering() == 'th':
        #     x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    # computes class activation function
    def grad_cam(self,input_model, input_data, cls, layer_name, fore_or_back=0):
        """GradCAM method for visualizing input saliency."""
        y_c = input_model.output[0, cls, fore_or_back]
        conv_output = input_model.get_layer(layer_name).output
        grads = K.gradients(y_c, conv_output)[0]
        # Normalize if necessary
        # grads = normalize(grads)
        gradient_function = K.function([input_model.input], [conv_output, grads])

        output, grads_val = gradient_function([input_data])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        output = np.transpose(output, axes=(1, 2, 0))
        grads_val = np.transpose(grads_val, axes=(1, 2, 0))

        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        # Process CAM
        cam = cv2.resize(cam, (204, 204), interpolation=cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 10e-20)
        return cam


    def get_mean_cam(self,mask,orig_img,layer_name, is_background):

        #prepare image
        patches_imgs_test, new_height, new_width, masks_test = get_data_segmenting_overlap(
            test_img_original=orig_img.astype('float'),
            Imgs_to_test=0,
            mean_image_path=self.mean_img_path,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            stride_height=1,
            stride_width=1,
            is_color=True
        )

        # CAM
        indices = np.nonzero(mask.flatten() > 0)
        nIdx = len(indices[0])
        cams = np.zeros((self.patch_height, self.patch_width, nIdx))
        range_idx = np.arange(0, nIdx, 2)
        #is_background = 0
        for i in range_idx:
            idx_test = indices[0][i]
            r, c = self.ind2sub(mask.shape, idx_test)
            gradcam = self.grad_cam(self.model, patches_imgs_test, idx_test, layer_name, is_background)
            cams[:, :, i] = gradcam

        final_cam = np.mean(cams, axis=(2))
        final_cam = final_cam / final_cam.max()

        norm = mpl.colors.Normalize(vmin=final_cam.min(), vmax=final_cam.max())
        cmap = cm.jet
        final_cam_rgb = cmap(final_cam)  # map "colors"
        final_cam_rgb = img_as_ubyte(final_cam_rgb)
        final_cam = final_cam_rgb[:, :, 0:3]

        alpha = 0.3
        output = cv2.addWeighted(final_cam, alpha, orig_img, 1 - alpha, 0)

        return output



def cam_test():
    config_file = '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/configuration_avid_slidenet_2class_204px.txt'
    layer_name = 'concatenate_3'

    root_dir = '/home/maryana/Projects/AVID_pipeline/python/UCSFSlideScan/convnet/gradCAM/imgs/AT100/img1/'
    file_pref = 'patch_82790376-6d05-11e9-81c1-484d7ede57b2'

    test_file = root_dir + file_pref + '.tif'
    mask_fore = root_dir + file_pref + '_mask.tif'
    mask_back = root_dir + file_pref + '_back_mask.tif'

    # Foreground mask
    fore_mask = io.imread(mask_fore)
    if fore_mask.ndim > 2:
        fore_mask = fore_mask[..., 0]

    # Background mask
    back_mask = io.imread(mask_back)
    if back_mask.ndim > 2:
        back_mask = back_mask[..., 0]

    # image
    orig_img = io.imread(test_file)


    grad_cam = GradCAM(config_file)
    cam_fore = grad_cam.get_mean_cam(fore_mask,orig_img,layer_name,0)
    plt.imshow(cam_fore)

    cam_back = grad_cam.get_mean_cam(back_mask, orig_img, layer_name,1)
    plt.imshow(cam_back)
    pass


if __name__ == '__main__':
    cam_test()

