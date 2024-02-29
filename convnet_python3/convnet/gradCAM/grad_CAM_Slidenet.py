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

def build_model():
    model = model_from_json(open(
        '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/models/AT100_slidenet_architecture.json').read())
    model.load_weights(
        '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/models/AT100_slidenet_best_weights.h5')
    return model

def build_guided_model():
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    grads_val = grads_val[0, ...]
    grads_val = np.transpose(grads_val, axes=(1, 2, 0))

    return grads_val

def grad_cam(input_model, input_data, cls, layer_name, fore_or_back = 0):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls, fore_or_back]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([input_data])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    output = np.transpose(output,axes=(1,2,0))
    grads_val = np.transpose(grads_val,axes=(1,2,0))

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # cam2 = np.zeros(cam.shape)
    # nW = weights.shape[0]
    # for i in range(nW):
    #     cam2 += weights[i] * output[:,:,i]


    # Process CAM
    cam = cv2.resize(cam, (204, 204), interpolation=cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 10e-20)
    return cam

def deprocess_image(x):
    """Same normalization as in:
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


mean_img_path = '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/training/mean_image.npy'
layer_name = 'concatenate_3'

root_dir = 'imgs/AT100/img3/'
file_pref = 'patch_c45a7447-6d05-11e9-81c1-484d7ede57b2'

test_file = root_dir + file_pref + '.tif'
test_file_perturb = root_dir + file_pref + '_perturb.tif'
mask_fore = root_dir + file_pref + '_mask.tif'
mask_back = root_dir + file_pref + '_back_mask.tif'
cam_file_fore = root_dir + 'cam_fore.tif'
cam_file_back = root_dir + 'cam_back.tif'
cam_file_fore_pert = root_dir + 'cam_fore_perturb.tif'
cam_file_back_pert = root_dir + 'cam_back_perturb.tif'
cam_ref_fore = root_dir + 'cam_ref_fore.tif'
cam_ref_back = root_dir + 'cam_ref_back.tif'
img_seg_file = root_dir + 'img_segmented.tif'
img_seg_file_pert = root_dir + 'img_segmented_perturb.tif'
mosaic_file = root_dir + file_pref +' _mosaic.tif'

prob_seg = root_dir + file_pref + '_prob.npy'
seg_mask_file = root_dir + file_pref + '_seg.tif'

#create image mosaic
mosaic = np.ones((408,1020,3),dtype='uint8')

# Load the saved model
model = build_model()
model.summary()

orig_img = io.imread(test_file)
patches_imgs_test, new_height, new_width, masks_test = get_data_segmenting_overlap(
    test_img_original=orig_img.astype('float'),
    Imgs_to_test=0,
    mean_image_path= mean_img_path,
    patch_height=204,
    patch_width=204,
    stride_height=1,
    stride_width=1,
    is_color=True
)

mosaic[0:204,0:204,:] = orig_img[:,:,:]

#guided_model = build_guided_model()

# Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=1, verbose=2)
pred_patches = pred_to_imgs(predictions, 200, 200, "original")
pred_patches = pred_patches[0,0,...]
pred_mask = pred_patches > 0.7
pred_mask = cv2.resize((pred_mask*255).astype('uint8'), (204, 204), interpolation=cv2.INTER_LINEAR)
#pred_mask[pred_mask < 255] = 0

io.imsave(seg_mask_file,pred_mask)
np.save(prob_seg,pred_patches)

exit()

perim = bwperim(pred_mask)
seg_img = imoverlay(orig_img,perim,[0,1,0])
cv2.imwrite(img_seg_file,cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))

mosaic[0:204,204:408,:] = seg_img[:,:,:]

#
# Foreground CAM
#

#Foreground mask
mask = io.imread(mask_fore)
if mask.ndim > 2:
    mask = mask[...,0]
overlay_mask = imoverlay(orig_img,mask,[1,0,0])
cv2.imwrite(cam_ref_fore,cv2.cvtColor(overlay_mask, cv2.COLOR_RGB2BGR))

mosaic[0:204,408:612,:] = overlay_mask[:,:,:]

#CAM
indices = np.nonzero(mask.flatten() > 0)
#indices2 = np.nonzero(mask > 0)
nIdx = len(indices[0])
cams = np.zeros((204,204,nIdx))
range_idx = np.arange(0,nIdx,2)
is_background = 0
for i in range_idx:
    idx_test = indices[0][i]
    r,c = ind2sub(mask.shape,idx_test)
    gradcam = grad_cam(model,patches_imgs_test,idx_test,layer_name,is_background)
    cams[:,:,i] = gradcam

final_cam = np.mean(cams,axis=(2))
final_cam = final_cam/final_cam.max()

norm = mpl.colors.Normalize(vmin=final_cam.min(), vmax=final_cam.max())
cmap = cm.jet
final_cam_rgb = cmap(final_cam) #map "colors"
final_cam_rgb = img_as_ubyte(final_cam_rgb)
final_cam = final_cam_rgb[:,:,0:3]

alpha = 0.3
output = cv2.addWeighted(final_cam, alpha, orig_img, 1 - alpha, 0)
cv2.imwrite(cam_file_fore,cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

mosaic[0:204,612:816,:] = output[:,:,:]

#
# Background CAM
#

#background mask
mask = io.imread(mask_back)
if mask.ndim > 2:
    mask = mask[...,0]
overlay_mask = imoverlay(orig_img,mask,[1,0,0])
cv2.imwrite(cam_ref_back,cv2.cvtColor(overlay_mask, cv2.COLOR_RGB2BGR))

mosaic[0:204,816:1020,:] = overlay_mask[:,:,:]

#CAM
indices = np.nonzero(mask.flatten() > 0)
#indices2 = np.nonzero(mask > 0)
nIdx = len(indices[0])
cams = np.zeros((204,204,nIdx))
range_idx = np.arange(0,nIdx,2)
is_background = 1
for i in range_idx:
    idx_test = indices[0][i]
    r,c = ind2sub(mask.shape,idx_test)
    gradcam = grad_cam(model,patches_imgs_test,idx_test,layer_name,is_background)
    cams[:,:,i] = gradcam

final_cam = np.mean(cams,axis=(2))
final_cam = final_cam/final_cam.max()

norm = mpl.colors.Normalize(vmin=final_cam.min(), vmax=final_cam.max())
cmap = cm.jet
final_cam_rgb = cmap(final_cam) #map "colors"
final_cam_rgb = img_as_ubyte(final_cam_rgb)
final_cam = final_cam_rgb[:,:,0:3]

alpha = 0.3
output = cv2.addWeighted(final_cam, alpha, orig_img, 1 - alpha, 0)
cv2.imwrite(cam_file_back,cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

mosaic[204:408,0:204,:] = output[:,:,:]


######
# Test perturbation
######


orig_img = io.imread(test_file_perturb)
mosaic[204:408,204:408,:] = orig_img[:,:,:]

patches_imgs_test, new_height, new_width, masks_test = get_data_segmenting_overlap(
    test_img_original=orig_img.astype('float'),
    Imgs_to_test=0,
    mean_image_path= mean_img_path,
    patch_height=204,
    patch_width=204,
    stride_height=1,
    stride_width=1,
    is_color=True
)

#guided_model = build_guided_model()

# Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=1, verbose=2)
pred_patches = pred_to_imgs(predictions, 200, 200, "original")
pred_patches = pred_patches[0,0,...]
pred_mask = pred_patches > 0.7
pred_mask = cv2.resize((pred_mask*255).astype('uint8'), (204, 204), interpolation=cv2.INTER_LINEAR)
perim = bwperim(pred_mask)
seg_img = imoverlay(orig_img,perim,[0,1,0])
cv2.imwrite(img_seg_file_pert,cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
mosaic[204:408,408:612,:] = seg_img[:,:,:]


#
# Foreground CAM
#

#Foreground mask
mask = io.imread(mask_fore)
if mask.ndim > 2:
   mask = mask[...,0]
#overlay_mask = imoverlay(orig_img,mask,[1,0,0])
#cv2.imwrite(cam_ref_fore,cv2.cvtColor(overlay_mask, cv2.COLOR_RGB2BGR))

#CAM
indices = np.nonzero(mask.flatten() > 0)
#indices2 = np.nonzero(mask > 0)
nIdx = len(indices[0])
cams = np.zeros((204,204,nIdx))
range_idx = np.arange(0,nIdx,2)
is_background = 0
for i in range_idx:
    idx_test = indices[0][i]
    r,c = ind2sub(mask.shape,idx_test)
    gradcam = grad_cam(model,patches_imgs_test,idx_test,layer_name,is_background)
    cams[:,:,i] = gradcam

final_cam = np.mean(cams,axis=(2))
final_cam = final_cam/final_cam.max()

norm = mpl.colors.Normalize(vmin=final_cam.min(), vmax=final_cam.max())
cmap = cm.jet
final_cam_rgb = cmap(final_cam) #map "colors"
final_cam_rgb = img_as_ubyte(final_cam_rgb)
final_cam = final_cam_rgb[:,:,0:3]

alpha = 0.3
output = cv2.addWeighted(final_cam, alpha, orig_img, 1 - alpha, 0)
cv2.imwrite(cam_file_fore_pert,cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

mosaic[204:408,612:816,:] = output[:,:,:]

#
# Background CAM
#

#Foreground mask
mask = io.imread(mask_back)
if mask.ndim > 2:
    mask = mask[...,0]
# overlay_mask = imoverlay(orig_img,mask,[1,0,0])
# cv2.imwrite(cam_ref_back,cv2.cvtColor(overlay_mask, cv2.COLOR_RGB2BGR))

#CAM
indices = np.nonzero(mask.flatten() > 0)
#indices2 = np.nonzero(mask > 0)
nIdx = len(indices[0])
cams = np.zeros((204,204,nIdx))
range_idx = np.arange(0,nIdx,2)
is_background = 1
for i in range_idx:
    idx_test = indices[0][i]
    r,c = ind2sub(mask.shape,idx_test)
    gradcam = grad_cam(model,patches_imgs_test,idx_test,layer_name,is_background)
    cams[:,:,i] = gradcam

final_cam = np.mean(cams,axis=(2))
final_cam = final_cam/final_cam.max()

norm = mpl.colors.Normalize(vmin=final_cam.min(), vmax=final_cam.max())
cmap = cm.jet
final_cam_rgb = cmap(final_cam) #map "colors"
final_cam_rgb = img_as_ubyte(final_cam_rgb)
final_cam = final_cam_rgb[:,:,0:3]

alpha = 0.3
output = cv2.addWeighted(final_cam, alpha, orig_img, 1 - alpha, 0)
cv2.imwrite(cam_file_back_pert,cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

mosaic[204:408,816:1020,:] = output[:,:,:]

cv2.imwrite(mosaic_file,cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))



