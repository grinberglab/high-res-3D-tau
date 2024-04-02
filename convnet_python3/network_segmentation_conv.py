import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#Python
import configparser as cparser
from keras.models import model_from_json
import sys
from convnet.util.help_functions import *
from convnet.util.extract_patches import recompone_overlap,get_data_segmenting_overlap

import fnmatch
from misc.imoverlay import imoverlay as imoverlay
import skimage.io as io
import glob
import matplotlib.pyplot as plt
import cv2

import time
from scipy import stats

class Segmentation:

    def __init__(self, config_file):
        self.TISSUE_THRESH = 0.01

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


    # count the number of pixels that are not background
    def get_num_pix_tissue(self, img):  # assumes RGB image
        tmp_img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        tmp_nnz_b = tmp_img.flatten().nonzero()
        nnz_b = float(len(tmp_nnz_b[0]))  # number of non-zero pixel in img
        return nnz_b

    # get a folder list inside _root_dir_
    def get_folder_list(self, root_dir):
        folder_list = []

        for root, dir, files in os.walk(root_dir):
            if fnmatch.fnmatch(root, '*heat_map'):
                folder_list.append(root)

        return folder_list


    def run_segmentation(self, root_dir):
        nError = 0

        dir_list = self.get_folder_list(root_dir)
        for folder in dir_list:

            # check if tiles folder exists
            tiles_dir = os.path.join(folder, 'seg_tiles_0054')
            if not os.path.exists(tiles_dir):
                print('Error: tiles folder {} does not exist.'.format(tiles_dir))
                continue

            # create output folder
            out_dir = os.path.join(folder, 'TAU_seg_tiles_0054')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
                
            # create converted folder
            conv_dir = os.path.join(folder, 'conv_seg_tiles_0054')
            if not os.path.exists(conv_dir):
                os.mkdir(conv_dir)

            print('*** Processing files in folder {}'.format(folder))

            # get a list of tif files
            files = glob.glob(os.path.join(tiles_dir, '*.tif'))
            nTotal = len(files)
            print('### {} tile(s) to segment.### '.format(nTotal))

            for fname in files:
                basename = os.path.basename(fname)
                out_name_seg = os.path.join(out_dir, basename[0:-4] + '_mask.tif')
                out_name_prob = os.path.join(out_dir, basename[0:-4] + '_prob.npy')
                
                # orig_name_npy_red = os.path.join(tiles_dir, basename[0:-4] + '_red.npy')
                # orig_name_npy_green = os.path.join(tiles_dir, basename[0:-4] + '_green.npy')
                # orig_name_npy_blue = os.path.join(tiles_dir, basename[0:-4] + '_blue.npy')
                orig_name_npy_conv = os.path.join(conv_dir, basename[0:-4] + '_conv.tif')

                test_imgs_original = os.path.join(tiles_dir, basename)
                print('Segmenting image {}.'.format(test_imgs_original))

                # Load the data and divide in patches
                try:
                    # load image to segment
                    orig_img = io.imread(test_imgs_original)
                except:
                    nError += 1
                    print("Error opening file {}".format(test_imgs_original))
                    continue
                
                print(orig_img.shape)
                # np.save(orig_name_npy_red, orig_img[:,:,0])
                # np.save(orig_name_npy_green, orig_img[:,:,1])
                # np.save(orig_name_npy_blue, orig_img[:,:,2])
                
                # conv_0 = 140 - (orig_img[:,:,0] / 2)
                # conv_1 = 120 - (orig_img[:,:,1] / 2)
                # conv_2 = 90 - (orig_img[:,:,2] / 2)
                x = orig_img[:,:,0] == 0
                conv_0 = orig_img[:,:,0]
                for ix, iy in np.ndindex(conv_0.shape):
                    if conv_0[ix, iy] >= 100:
                        conv_0[ix, iy] = conv_0[ix, iy]/2 + 155
                    else:
                        conv_0[ix, iy] = conv_0[ix, iy] + 155
                conv_0[x] = 0
                y = orig_img[:,:,1] == 0
                conv_1 = orig_img[:,:,1] + 90
                conv_1[y] = 0
                z = orig_img[:,:,2] == 0
                conv_2 = orig_img[:,:,2] + 50
                conv_2[z] = 0
                # conv_output = np.concatenate((conv_0, conv_1, conv_2), axis = 1)
                conv_output = np.stack((conv_0, conv_1, conv_2), axis = 2).astype('uint8')
                print(conv_output.shape)
                
                io.imsave(orig_name_npy_conv, conv_output)
                orig_img = conv_output

                # check if image has enough tissue
                npix_tissue = self.get_num_pix_tissue(orig_img)
                percent_tissue = npix_tissue / (orig_img.shape[0] * orig_img.shape[1])
                if percent_tissue < self.TISSUE_THRESH:
                    print('Image has too little tissue. Skipping.')
                    continue

                # pad sides
                orig_img_pad = pad_image(orig_img.copy(), self.patch_height, self.patch_width)

                # original tiles are 1024x1024
                # break image into smaller patches of 204x204 (network input size)
                patches_imgs_test, new_height, new_width, masks_test = get_data_segmenting_overlap(
                    test_img_original = orig_img_pad.astype('float'),  # image path to segment
                    Imgs_to_test = self.imgs_to_test,
                    mean_image_path = self.mean_img_path,
                    patch_height = self.patch_height,
                    patch_width = self.patch_width,
                    stride_height = self.stride_height,
                    stride_width = self.stride_width,
                    is_color = True
                )

                # calculate the predictions
                start = time.perf_counter()
                predictions = self.model.predict(patches_imgs_test, batch_size=32, verbose=2)
                end = time.perf_counter()
                print("**Time per image: {} ".format((end - start) / 32))
                print("predicted images size :")
                print(predictions.shape)

                # convert the prediction arrays in corresponding images
                pred_patches = pred_to_imgs(predictions, self.mask_dim[0], self.mask_dim[1], "original")

                new_pred_patches = np.zeros((pred_patches.shape[0], 1, self.patch_height, self.patch_width))
                nP = pred_patches.shape[0]

                # network output is 200x200, resize to original dimensions
                for p in range(nP):
                    tmp = pred_patches[p, 0, ...]
                    tmp = cv2.resize(tmp, (self.patch_height,self. patch_width), interpolation=cv2.INTER_NEAREST)
                    new_pred_patches[p, 0, ...] = tmp


                pred_imgs = recompone_overlap(new_pred_patches, new_height, new_width, self.stride_height,
                                                  self.stride_width)  # predictions
                img = pred_imgs[0, 0, ...]  # get matrix for Tau prediction

                # remove padding 1
                pad_r1 = new_height - orig_img_pad.shape[0]
                pad_c1 = new_width - orig_img_pad.shape[1]
                img = img[0:img.shape[0] - pad_r1, 0:img.shape[1] - pad_c1, ...]

                # remove padding 2
                img = img[self.patch_height:img.shape[0] - self.patch_height, self.patch_width:img.shape[1] - self.patch_width, ...]

                # threshold
                mask = img > 0.7
                # mask = img > 0.5  # threshold class probabilities
                #mask = (img < 0.9) & (img > 0.4)
                #img_copy = img.copy()
                #img_copy[mask] = 0

                print('Saving probability file {}'.format(out_name_prob))
                np.save(out_name_prob, img)

                # mask out background just in case
                mask_bkg = orig_img[..., 0] < 1.
                mask[mask_bkg == True] = False

                print('Saving {}'.format(out_name_seg))
                io.imsave(out_name_seg, (mask * 255).astype('uint8'))

            print("Segmentation ended with {} errors".format(nError))


def main():
    if len(sys.argv) != 3:
        print('Usage: network_segmentation <root_dir> <config_file.txt>')
        exit()

    root_dir = str(sys.argv[1])
    config_path = str(sys.argv[2])

    segmentation = Segmentation(config_path)
    segmentation.run_segmentation(root_dir)


if __name__ == '__main__':
    main()
