from keras.applications import ResNet50
from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations
import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam

from keras import backend
backend.set_image_dim_ordering('tf') #force channels_last


# Build the ResNet50 network with ImageNet weights
model = ResNet50(weights='imagenet', include_top=True)

# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'fc1000')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)


plt.rcParams['figure.figsize'] = (18, 6)

img1 = utils.load_img('/home/maryana/storage/Posdoc/keras-vis/examples/vggnet/images/ouzel1.jpg', target_size=(224, 224))
img2 = utils.load_img('/home/maryana/storage/Posdoc/keras-vis/examples/vggnet/images/ouzel2.jpg', target_size=(224, 224))

f, ax = plt.subplots(1, 2)
ax[0].imshow(img1)
ax[1].imshow(img2)


# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'fc1000')

f, ax = plt.subplots(1, 2)
for i, img in enumerate([img1, img2]):
    # 20 is the imagenet index corresponding to `ouzel`
    grads = visualize_saliency(model, layer_idx, filter_indices=20, seed_input=img)

    # visualize grads as heatmap
    ax[i].imshow(grads, cmap='jet')


for modifier in ['guided', 'relu']:
    f, ax = plt.subplots(1, 2)
    plt.suptitle(modifier)
    for i, img in enumerate([img1, img2]):
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_saliency(model, layer_idx, filter_indices=20,
                                   seed_input=img, backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        ax[i].imshow(grads, cmap='jet')

    plt.show()


penultimate_layer = utils.find_layer_idx(model, 'res5c_branch2c')

for modifier in [None, 'guided', 'relu']:
    f, ax = plt.subplots(1, 2)
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img1, img2]):
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_cam(model, layer_idx, filter_indices=20,
                              seed_input=img, penultimate_layer_idx=penultimate_layer,
                              backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        ax[i].imshow(overlay(jet_heatmap, img))