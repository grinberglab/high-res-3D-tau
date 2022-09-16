import os
import sys
import ConfigParser
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations
import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam
from keras.models import model_from_json
from convnet.util.pre_processing import preproc_color
import skimage.io as io

config_file = '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/configuration_avid_slidenet_2class_204px.txt'

# Load the saved model
config = ConfigParser.RawConfigParser()
config.read(config_file)
path_project = config.get('data paths', 'path_project')
path_model = os.path.join(path_project, config.get('data paths', 'path_model'))
name_experiment = config.get('experiment name', 'name')

model = model_from_json(open(os.path.join(path_model, name_experiment + '_architecture.json')).read())
model.load_weights(os.path.join(path_model, name_experiment + '_best_weights.h5'))

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'activation_1')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

img_file = '/home/maryana/Projects/LargeSlideScan/python/UCSFSlideScan/tests/patch_test.tif'
mean_img_path = os.path.join(path_project, config.get('data paths', 'mean_image'))

test_img_orig = io.imread(img_file)

test_img = test_img_orig.astype('float')
test_img = np.transpose(test_img, axes=(2, 0, 1))
test_img = test_img.reshape([1, test_img.shape[0], test_img.shape[1], test_img.shape[2]])
test_img = preproc_color(test_img,mean_img_path)

modifier = 'guided'
penultimate_layer = utils.find_layer_idx(model, 'conv2d_14')
grads = visualize_cam(model, layer_idx, filter_indices=0,
                      seed_input=test_img, penultimate_layer_idx=penultimate_layer,
                      backprop_modifier=modifier)

jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
f, ax = plt.subplots(1, 2)
ax[0].imshow(jet_heatmap)
ax[1].imshow(test_img_orig)
pass

