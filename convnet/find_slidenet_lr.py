###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import configparser
import sys
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from convnet.net.TauImageGenerator import TauImageGenerator
import convnet.net.Slidenet
from convnet.util.lr_finder import LRFinder


def get_data(nBatches, train_imgs_dir, train_masks_dir, mean_img_path, img_dim, mask_dim, nClasses, batch_size):

    tau_gen = TauImageGenerator('find_lr_gen', train_imgs_dir, train_masks_dir, mean_img_path, img_dim, mask_dim, nClasses, batch_size, do_augmentation=True, augment_percent=0.50)
    batch_gen = tau_gen.get_batch()

    X_train = [] #images
    Y_train = [] #masks
    for i in range(nBatches):

        x,y = next(batch_gen)

        if X_train == []:
            X_train = x
        else:
            X_train = np.concatenate((X_train,x),axis=0)
        if Y_train == []:
            Y_train = y
        else:
            Y_train = np.concatenate((Y_train,y),axis=0)

    return X_train,Y_train



def run_training(conf_path):

    config = configparser.RawConfigParser()
    config.read(conf_path)

    #Experiment name
    name_experiment = config.get('experiment name', 'name')
    #training settings
    N_epochs = int(config.get('training settings', 'N_epochs'))
    batch_size = int(config.get('training settings', 'batch_size'))
    path_project = config.get('data paths', 'path_project')
    path_model = os.path.join(path_project, config.get('data paths', 'path_model'))

    train_imgs_dir = os.path.join(path_project,config.get('data paths', 'train_imgs_original'))
    train_masks_dir = os.path.join(path_project,config.get('data paths', 'train_groundTruth'))
    test_imgs_dir = os.path.join(path_project,config.get('data paths', 'test_imgs_original'))
    test_masks_dir = os.path.join(path_project,config.get('data paths', 'test_groundTruth'))
    mean_img_path = os.path.join(path_project, config.get('data paths', 'mean_image'))
    train_log = os.path.join(path_project, config.get('data paths', 'train_log'))

    n_ch = int(config.get('data attributes','num_channels'))
    patch_height = int(config.get('data attributes','patch_height'))
    patch_width = int(config.get('data attributes','patch_width'))
    img_dim = (patch_height,patch_width,n_ch)
    nClasses = int(config.get('data attributes','num_classes'))

    patch_height = 204
    patch_width = 204
    batch_size = 32
    img_dim = (patch_height, patch_width, n_ch)
    mask_dim = (200,200)

    #model = Slidenet.get_slidenet2(n_ch, patch_height, patch_width)  #the model
    model = convnet.net.Slidenet.get_slidenet2(n_ch, 204, 204)

    print('Creating dataset.')
    nBatches_ds = 60
    img_train,mask_train = get_data(nBatches_ds, train_imgs_dir, train_masks_dir, mean_img_path, img_dim, mask_dim, nClasses, batch_size)

    lr_finder = LRFinder(min_lr=1e-6,
                         max_lr=1e-1,
                         steps_per_epoch=np.ceil(img_train.shape[0] / float(nBatches_ds)),
                         epochs=60)
    model.fit(img_train, mask_train, callbacks=[lr_finder], epochs=60)

    lr_finder.plot_loss()
    pass

    # lr_finder = kf.LRFinder(model)
    # lr_finder.find(img_train, mask_train, start_lr=0.0001, end_lr=0.1, batch_size=batch_size, epochs=10)
    #
    # lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
    # lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))




def main():
    if len(sys.argv) != 2:
        print('Usage: lr_finder <config_file.txt>')
        exit()

    config_path = str(sys.argv[1])
    run_training(config_path)


if __name__ == '__main__':
    main()













