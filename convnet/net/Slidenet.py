from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
import numpy as np


#set random seed
from numpy.random import seed
seed(17)
from tensorflow import set_random_seed
set_random_seed(17)




# #Define the neural network
def get_slidenet(n_ch=3,patch_height=818,patch_width=818): #(3,818,818)
    inputs = Input(shape=(n_ch, patch_height, patch_width))

    conv1 = Conv2D(16, (3, 3), activation='relu',data_format='channels_first')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', data_format='channels_first')(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu',data_format='channels_first')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)

    up1 = UpSampling2D(size=(2,2))(conv5)
    up1 = concatenate([conv4,up1],axis=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', data_format='channels_first')(up1)
    conv6 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(conv6)

    up2 = UpSampling2D(size=(2,2))(conv6)
    crop_conv3 = Cropping2D(cropping=((12, 12), (12, 12)))(conv3)
    up2 = concatenate([crop_conv3,up2],axis=1)
    conv7 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(up2)
    conv7 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(conv7)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    crop_conv2 = Cropping2D(cropping=((32, 32), (32, 32)))(conv2)
    up3 = concatenate([crop_conv2,up3],axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(up3)
    conv8 = Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(conv8)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    crop_conv1 = Cropping2D(cropping=((76, 76), (76, 76)))(conv1)
    up4 = concatenate([crop_conv1,up4],axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(up4)
    conv9 = Conv2D(16, (3, 3), activation='relu', data_format='channels_first')(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', data_format='channels_first')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='relu', data_format='channels_first')(conv9)
    conv10 = Dropout(0.2)(conv10)
    conv10 = core.Reshape((2, 654 * 654))(conv10)
    conv10 = core.Permute((2, 1))(conv10)
    softmax = core.Activation('softmax')(conv10)
    model = Model(input=inputs, output=softmax)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    adam = Adam(lr=3e-4)
    model.compile(optimizer=SGD(lr=0.021), loss='categorical_crossentropy',metrics=['accuracy'])

    model.summary()

    return model


def get_slidenet2(n_ch=3,patch_height=204,patch_width=204):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    conv1 = Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.1)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.1)(pool3)

    conv4 = Conv2D(256, (1, 1), activation='relu', data_format='channels_first')(pool3)
    conv4 = Conv2D(256, (1, 1), activation='relu', data_format='channels_first')(conv4)
    up4 = UpSampling2D(size=(2, 2))(conv4)
    up4 = concatenate([conv3, up4], axis=1)
    up4 = Dropout(0.1)(up4)

    conv5 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(up4)
    conv5 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first')(conv5)
    up5 = UpSampling2D(size=(2, 2))(conv5)
    crop_conv6 = Cropping2D(cropping=((8, 8), (8, 8)))(conv2)
    up6 = concatenate([crop_conv6, up5], axis=1)
    up6 = Dropout(0.1)(up6)

    conv6 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(conv6)
    up7 = UpSampling2D(size=(3, 3))(conv6)
    crop_conv7 = Cropping2D(cropping=((14, 14), (14, 14)))(up7)
    up7 = concatenate([conv1, crop_conv7], axis=1)
    up7 = Dropout(0.1)(up7)

    conv7 = Conv2D(32, (1, 1), activation='relu', data_format='channels_first')(up7)
    conv7 = Conv2D(2, (1, 1), activation='relu', data_format='channels_first')(conv7)
    conv7 = Dropout(0.1)(conv7)

    conv7 = core.Reshape((2, 200 * 200))(conv7)
    conv7 = core.Permute((2, 1))(conv7)
    softmax = core.Activation('softmax')(conv7)
    #softmax = core.Activation('sigmoid')(conv7)
    model = Model(input=inputs, output=softmax)

    #sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = False)
    #adam = Adam(lr=0.0021) #AT100
    #adam = Adam(lr=0.002701)  # AT8
    adam = Adam(lr=0.005) #MC1

    #sgd = SGD(lr=0.005)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
    #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy', 'categorical_accuracy', dice_coef])

    model.summary()

    return model

def dice_coef(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)






