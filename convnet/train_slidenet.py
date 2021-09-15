###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
import ConfigParser
import sys
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from convnet.net.TauImageGenerator import TauImageGenerator
import convnet.net.Slidenet
from convnet.util.clr_callback import *



def lr_scheduler(epoch, initial_lr=0.1, decay_factor=0.75, step_size=10):
    return initial_lr * (decay_factor ** np.floor(epoch / step_size))


def run_training(conf_path):

    config = ConfigParser.RawConfigParser()
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
    print "Check: final output of the network:"
    print model.output_shape
    json_string = model.to_json()

    model_file = os.path.join(path_model,name_experiment + '_architecture.json')
    best_weights_file = os.path.join(path_model,name_experiment + 'weights_{epoch:03d}_{val_loss:.4f}.h5')
    last_weights_files = os.path.join(path_model,name_experiment + '_last_weights.h5')

    open(model_file, 'w').write(json_string)

    checkpointer = ModelCheckpoint(filepath= best_weights_file, verbose=1, monitor='val_loss', mode='auto', save_best_only=False) #save at each epoch if the validation decreases
    tensorboard = TensorBoard(log_dir=train_log, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)

    #train_gen = TauImageGenerator('train_gen',train_imgs_dir,train_masks_dir,mean_img_path,img_dim,mask_dim,nClasses,batch_size,do_augmentation=False,augment_percent=0.40,resize_mask=[],class_weights=(0.6,0.2))
    train_gen = TauImageGenerator('train_gen', train_imgs_dir, train_masks_dir, mean_img_path, img_dim, mask_dim, nClasses, batch_size, do_augmentation=True, augment_percent=0.40)
    test_gen = TauImageGenerator('test_gen',test_imgs_dir, test_masks_dir, mean_img_path, img_dim, mask_dim, nClasses, batch_size, do_augmentation=False, augment_percent=0.40)
    #test_gen = TauImageGenerator('test_gen',train_imgs_dir, train_masks_dir, mean_img_path, img_dim, mask_dim, nClasses, batch_size,do_augmentation=False,augment_percent=0.40)

    sample_weights = np.zeros((200*200,2))
    sample_weights[...,0] = 0.8
    sample_weights[...,1] = 0.2


    #scheduler = LearningRateScheduler(lr_scheduler, verbose=1)

    #clr_triangular = CyclicLR(mode='triangular', step_size=100, max_lr=0.006)


    #model.fit(patches_imgs_train, patches_masks_train, nb_epoch=N_epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.1, callbacks=[checkpointer,tensorboard])
    model.fit_generator(generator=train_gen.get_batch(),
                        validation_data=test_gen.get_batch(),
                        steps_per_epoch=train_gen.__len__(),
                        validation_steps=test_gen.__len__(),
                        epochs=100,
                        verbose=1,
                        callbacks = [checkpointer, tensorboard])

    # plt.xlabel('Training Iterations')
    # plt.ylabel('Learning Rate')
    # plt.title("CLR - 'triangular' Policy")
    # plt.plot(clr_triangular.history['iterations'], clr_triangular.history['lr'])

    model.save_weights(last_weights_files, overwrite=True)



def main():
    if len(sys.argv) != 2:
        print('Usage: run_prediction <config_file.txt>')
        exit()

    config_path = str(sys.argv[1])
    run_training(config_path)


if __name__ == '__main__':
    main()













