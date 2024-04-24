import numpy as np
import configparser
import sys
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from convnet.net.TauImageGenerator import TauImageGenerator
import convnet.net.Slidenet
from convnet.util.lr_finder import LRFinder
from convnet.net.slidenet_factory import SlidenetFactory

import tensorflow as tf
tf.config.list_physical_devices('GPU')


class NetworkTrainer:

    def __init__(self,config_file):
        config = configparser.RawConfigParser()
        config.read(config_file)

        # Experiment name
        self.name_experiment = config.get('experiment name', 'name')

        # training settings
        self.N_epochs = int(config.get('training settings', 'N_epochs'))
        self.batch_size = int(config.get('training settings', 'batch_size'))
        self.path_project = config.get('data paths', 'path_project')
        self.path_model = os.path.join(self.path_project, config.get('data paths', 'path_model'))
        self.n_ch = int(config.get('data attributes', 'num_channels'))
        self.patch_height = int(config.get('data attributes', 'patch_height'))
        self.patch_width = int(config.get('data attributes', 'patch_width'))
        self.img_dim = (self.patch_height, self.patch_width, self.n_ch)
        self.nClasses = int(config.get('data attributes', 'num_classes'))
        self.mask_height = int(config.get('data attributes', 'mask_height'))
        self.mask_width = int(config.get('data attributes', 'mask_width'))
        self.mask_dim = (self.mask_height,self.mask_width)

        # relevant paths
        self.train_imgs_dir = os.path.join(self.path_project, config.get('data paths', 'train_imgs_original'))
        self.train_masks_dir = os.path.join(self.path_project, config.get('data paths', 'train_groundTruth'))
        self.test_imgs_dir = os.path.join(self.path_project, config.get('data paths', 'test_imgs_original'))
        self.test_masks_dir = os.path.join(self.path_project, config.get('data paths', 'test_groundTruth'))
        self.mean_img_path = os.path.join(self.path_project, config.get('data paths', 'mean_image'))
        self.train_log = os.path.join(self.path_project, config.get('data paths', 'train_log'))

        #get network model
        self.net_name = config.get('experiment name','network')

        # patch_height = 204
        # patch_width = 204
        # batch_size = 32
        # img_dim = (patch_height, patch_width, n_ch)
        # mask_dim = (200, 200)
        # mask_dim = (200, 200)


    def run_training(self,l_rate):

        # model = Slidenet.get_slidenet2(n_ch, patch_height, patch_width)  #the model
        #model = convnet.net.Slidenet.get_slidenet2(self.n_ch, self.patch_height, self.patch_width)
        cnn_factory = SlidenetFactory()
        model = cnn_factory.get_model(self.net_name,self.n_ch, self.patch_height, self.patch_width, l_rate)

        print("Check: final output of the network:")
        print(model.output_shape)
        json_string = model.to_json()

        model_file = os.path.join(self.path_model, self.name_experiment + '_architecture.json')
        best_weights_file = os.path.join(self.path_model, self.name_experiment + 'weights_{epoch:03d}_{val_loss:.4f}.h5')
        last_weights_files = os.path.join(self.path_model, self.name_experiment + '_last_weights.h5')

        open(model_file, 'w').write(json_string)

        # Keras callbacks
        checkpointer = ModelCheckpoint(filepath=best_weights_file, verbose=1, monitor='val_loss', mode='auto',
                                       save_best_only=False)  # save at each epoch if the validation decreases
        tensorboard = TensorBoard(log_dir=self.train_log, histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                  write_grads=False,
                                  write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                  embeddings_metadata=None)

        # image generators
        train_gen = TauImageGenerator('train_gen', self.train_imgs_dir, self.train_masks_dir, self.mean_img_path,
                                      self.img_dim, self.mask_dim, self.nClasses, self.batch_size,
                                      do_augmentation=True, augment_percent=0.40)
        test_gen = TauImageGenerator('test_gen', self.test_imgs_dir, self.test_masks_dir, self.mean_img_path,
                                     self.img_dim, self.mask_dim, self.nClasses, self.batch_size,
                                     do_augmentation=False, augment_percent=0.40)

        model.fit(generator=train_gen.get_batch(),
                            validation_data=test_gen.get_batch(),
                            steps_per_epoch=train_gen.__len__(),
                            validation_steps=test_gen.__len__(),
                            #epochs=100,
                            epochs=self.N_epochs,
                            verbose=1,
                            callbacks=[checkpointer, tensorboard])

        model.save_weights(last_weights_files, overwrite=True)

    def get_data_lr(self,nBatches):

        tau_gen = TauImageGenerator('find_lr_gen', self.train_imgs_dir, self.train_masks_dir, self.mean_img_path,
                                    self.img_dim, self.mask_dim, self.nClasses, self.batch_size, do_augmentation=True,
                                    augment_percent=0.50)
        batch_gen = tau_gen.get_batch()

        X_train = []  # images
        Y_train = []  # masks
        for i in range(nBatches):
            x, y = next(batch_gen)
            if len(X_train) == 0:
                X_train = x
            else:
                X_train = np.concatenate((X_train, x), axis=0)
            if len(Y_train) == 0:
                Y_train = y
            else:
                Y_train = np.concatenate((Y_train, y), axis=0)
        return X_train, Y_train


    def run_find_lr(self):

        # model = Slidenet.get_slidenet2(n_ch, patch_height, patch_width)  #the model
        #model = convnet.net.Slidenet.get_slidenet2(n_ch, 204, 204)

        cnn_factory = SlidenetFactory()
        model = cnn_factory.get_model(self.net_name,self.n_ch, self.patch_height, self.patch_width)
    
        print('Creating dataset.')
        nBatches_ds = 100
        img_train, mask_train = self.get_data_lr(nBatches_ds)
        # img_train = tf.transpose(img_train, [0, 3, 1, 2])
        # mask_train = tf.transpose(mask_train, [0, 3, 1, 2])

        lr_finder = LRFinder(min_lr=1e-4,
                             max_lr=10,
                             steps_per_epoch=np.ceil(img_train.shape[0] / float(nBatches_ds)),
                             epochs=100)
        model.fit(img_train, mask_train, callbacks=[lr_finder], epochs=100)

        lr_finder.plot_loss()


        # lr_finder = kf.LRFinder(model)
        # lr_finder.find(img_train, mask_train, start_lr=0.0001, end_lr=0.1, batch_size=batch_size, epochs=10)
        #
        # lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
        # lr_finder.plot_loss_change(sma=20, n_skip_beginning=20, n_skip_end=5, y_lim=(-0.01, 0.01))


def main():
    if len(sys.argv) < 3:
        print('Usage: network_trainer <train|find_lr> <config_file.txt> <l_rate>')
        exit()

    config_path = str(sys.argv[2])
    flag = str(sys.argv[1])
    cnn_trainer = NetworkTrainer(config_path)

    if flag == 'train':
        l_rate = float(sys.argv[3])
        cnn_trainer.run_training(l_rate)
    elif flag == 'find_lr':
        cnn_trainer.run_find_lr()

if __name__ == '__main__':
    main()
