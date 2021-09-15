import skimage.io as io
import matplotlib.pyplot as plt
import keras.preprocessing.image as k_image
import glob
import convnet.util.pre_processing as pp
import os
import random
import numpy as np
import keras
import cv2

#class TauImageGenerator(keras.utils.Sequence):
class TauImageGenerator:
    'Generates data for Keras'
    def __init__(self,gen_name,images_dir,masks_dir,mean_img,img_dim,mask_dim,nClasses,batch_size,do_augmentation=False,augment_percent=0.40,resize_mask=[],class_weights=[]):

        self.name = gen_name
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        #self.mean_image_path = mean_img
        self.patch_rows = img_dim[0]
        self.patch_cols = img_dim[1]
        self.mask_rows = mask_dim[0]
        self.mask_cols = mask_dim[1]
        self.nChannels = img_dim[2]
        self.nClasses = nClasses
        self.batch_size = batch_size #total num files per batch, nBack+nFore
        self.mu = pp.load_mean_values(mean_img)

        print(self.mu[...])

        #get file names
        #self.back_img_list = glob.glob(os.path.join(images_dir, '*_1_*.tif'))
        self.fore_img_list = glob.glob(os.path.join(images_dir, '*_0_*.tif'))
        self.back_img_list = self.fore_img_list
        #self.back_img_list = []

        #shuffle arrays and balance data
        self.shuffle_files()
        self.balance_data()

        #get num of bkg and tau tiles
        self.nBack = len(self.back_img_list)
        self.nFore = len(self.fore_img_list)

        #augmentation stuff
        self.resize_mask = resize_mask
        self.do_aug = do_augmentation
        self.augment_percent = augment_percent
        self.kera_generator = KerasDataGenerator(rotation_range=40,width_shift_range=0,height_shift_range=0,horizontal_flip=True,vertical_flip=True,fill_mode='nearest')

        # ensure batch_size is even
        if self.batch_size % 2 != 0:
            print('Warning: batch size must be an even number. Adjusting.')
            self.batch_size += 1

        #compute num. files to augment
        if self.do_aug:
            nAug = np.floor((self.batch_size/2) * self.augment_percent)
            self.nFilesAug = nAug*2 #total num of files to augment (nFore + nBack)
        else:
            self.nFilesAug = 0

        #compute num. of epochs
        if not self.do_aug:
            self.nBatches = int(np.floor((self.nFore + self.nBack) / self.batch_size))
        else:
            new_batch_size = self.batch_size - self.nFilesAug
            self.nBatches = int(np.floor((self.nFore + self.nBack) / new_batch_size))

        #current files array index
        self.current = 0

        #class weights
        self.class_weights = []
        if len(class_weights) == self.nClasses:
            self.class_weights = class_weights
        else:
            print('Warning: class_weights must have the same number of components than nClasses. Ignoring weights.')


        print('{}: #batches: {}'.format(gen_name,self.nBatches))


    def balance_data(self):

        nFore = len(self.fore_img_list)
        nBack = len(self.back_img_list)

        if nFore < nBack:
            diff = nBack - nFore
            idx = np.random.choice(nFore,diff,replace=True)
            tmp_list = [self.fore_img_list[i] for i in idx]
            self.fore_img_list += tmp_list
        elif nFore > nBack:
            diff = nFore - nBack
            idx = np.random.choice(nBack,diff,replace=True)
            tmp_list = [self.back_img_list[i] for i in idx]
            self.back_img_list += tmp_list


    def shuffle_files(self):
        random.shuffle(self.fore_img_list)
        random.shuffle(self.back_img_list)

    def __len__(self):
        return self.nBatches
        #return 1000

    def on_epoch_end(self):
        print('Epoch ended')
        self.shuffle_files()
        self.current = 0

    def get_batch(self):

        while 1:
            X,Y,W = self.__getitem__(self.current)

            #print('  {}: Current: {}'.format(self.name, self.current))

            self.current += 1
            if self.current >= self.nBatches:
                self.on_epoch_end()

            if self.class_weights == []:
                yield X,Y
            else:
                yield X,Y,W

    def __getitem__(self, index):

        #print('     {} Current: {}'.format(self.name,self.current))

        if not self.do_aug: #don't do data augmentation
            begin = int(index*(self.batch_size/2))
            end = int(begin + (self.batch_size/2))
            tmp_img_list = self.fore_img_list[begin:end]
            tmp_img_list += self.back_img_list[begin:end]
            random.shuffle(tmp_img_list)
            # #not really used without augmentation
            # self.current = end

        else:
            nOrig = (self.batch_size / 2) - (self.nFilesAug / 2)  # num original foreground OR background files. Total original is 2*nOrig
            begin = int(index*nOrig)
            end = int(begin + nOrig)
            #get only original files
            # method '__data_generation()' will created augmentes images/masks
            tmp_img_list = self.fore_img_list[begin:end]
            tmp_img_list += self.back_img_list[begin:end]
            random.shuffle(tmp_img_list)

        x,y,w = self.__data_generation(tmp_img_list)

        #self.current += 1

        return x,y,w




    #load an entire batch and do augmentation if necessary
    def __data_generation(self,file_arr):
        sample_weights = []
        img_vol = np.empty((self.batch_size, self.nChannels, self.patch_rows, self.patch_cols))
        if self.resize_mask == []:
            mask_vol = np.empty((self.batch_size, self.mask_rows*self.mask_cols, self.nClasses))
            if self.class_weights != []:
                sample_weights = np.empty((self.batch_size, self.mask_rows*self.mask_cols))
        else:
            mask_vol = np.empty((self.batch_size, self.resize_mask[0]*self.resize_mask[1], self.nClasses))
            if self.class_weights != []:
                sample_weights = np.empty((self.batch_size, self.resize_mask[0]*self.resize_mask[1]))

        count = 0
        count_aug = 0
        for fi in file_arr:
            img_name = os.path.basename(fi)
            img_path = os.path.join(self.images_dir, img_name)

            tmp_name = img_name[5:-3]
            mask_name = 'patch_mask' + tmp_name +'npy'
            mask_path = os.path.join(self.masks_dir, mask_name)

            # get image
            img = io.imread(img_path)

            if img.shape[0] != self.patch_rows or img.shape[1] != self.patch_cols:
                print('Patch has the wrong size ({},{}). Skipping.'.format(img.shape[0],img.shape[1]))
                continue

            img = img.astype('float')
            img = self.preproc_color(img)
            img_bkp = img.copy()
            img = np.transpose(img, axes=(2, 0, 1))
            img_vol[count, ...] = img

            # get mask
            mask = np.load(mask_path)
            if self.resize_mask != []:
                mask = cv2.resize(mask,(self.resize_mask[1],self.resize_mask[0]),interpolation=cv2.INTER_NEAREST)
            mask = mask.astype('float')
            mask /= 255
            mask_bkp = mask.copy()
            mask = mask.reshape((mask.shape[0] * mask.shape[1], mask.shape[2])) # dim3 = 0 -> foreground; dim3 = 1 -> background
            mask_vol[count, ...] = mask

            #create weights
            if self.class_weights != []:
                weights = np.zeros((mask.shape[0]))
                for nC in range(self.nClasses):
                    mask_tmp = mask[...,nC] > 0
                    weights[mask_tmp] = self.class_weights[nC]
                sample_weights[count, ...] = weights


            #do augmentation if flag is set
            #number of augmented images is always less than or equal to the number of original files
            if self.do_aug:
                if count_aug < self.nFilesAug:
                    count += 1
                    img_aug,mask_aug = self.kera_generator.random_transform2(img_bkp,mask_bkp)
                    img_aug = np.transpose(img_aug, axes=(2, 0, 1))
                    mask_aug = mask_aug.reshape((mask_aug.shape[0] * mask_aug.shape[1], mask_aug.shape[2]))

                    img_vol[count, ...] = img_aug
                    mask_vol[count, ...] = mask_aug

                    # create weights
                    if self.class_weights != []:
                        weights = np.zeros((mask_aug.shape[0]))
                        for nC in range(self.nClasses):
                            mask_tmp = mask_aug[..., nC] > 0
                            weights[mask_tmp] = self.class_weights[nC]
                        sample_weights[count, ...] = weights

                    count_aug += 1

            count += 1

        return img_vol,mask_vol,sample_weights #return X,Y


    def preproc_color(self, data):

        muR = self.mu[0]
        muG = self.mu[1]
        muB = self.mu[2]

        R = data[...,0]
        G = data[...,1]
        B = data[...,2]
        R = R - muR
        G = G - muG
        B = B - muB

        data[...,0] = R
        data[...,1] = G
        data[...,2] = B

        data /= 255.

        return data




class KerasDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    def __init__(self,rotation_range=0., width_shift_range=0., height_shift_range=0., horizontal_flip = False, vertical_flip = False, fill_mode='nearest'):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode
        super(KerasDataGenerator,self).__init__(rotation_range=self.rotation_range,
                                                width_shift_range=self.width_shift_range,
                                                height_shift_range=self.height_shift_range,
                                                horizontal_flip=self.horizontal_flip,
                                                vertical_flip = self.vertical_flip,
                                                fill_mode=self.fill_mode)


    # modified from Keras
    # do augmentation on image and mask
    # only performs geometric tranforms (no color modification)
    def random_transform2(self, x, y, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image: [rows,cols,channels]
        # img_row_axis = self.row_axis - 1
        # img_col_axis = self.col_axis - 1
        # img_channel_axis = self.channel_axis - 1
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 2


        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            #transform image
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = k_image.transform_matrix_offset_center(transform_matrix, h, w)
            x = k_image.apply_transform(x, transform_matrix, img_channel_axis, fill_mode=self.fill_mode, cval=self.cval)
            y = k_image.apply_transform(y, transform_matrix, img_channel_axis, fill_mode=self.fill_mode, cval=self.cval)

        # if super(KerasDataGenerator,self).channel_shift_range != 0:
        #     x = k_image.random_channel_shift(x,super(KerasDataGenerator, self).channel_shift_range,img_channel_axis)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = k_image.flip_axis(x, img_col_axis)
                y = k_image.flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = k_image.flip_axis(x, img_row_axis)
                y = k_image.flip_axis(y, img_row_axis)

        #threshold mask to avoid interpolation artifacts
        y[y < 1] = 0

        return x,y


#test generator
def main():
    images_dir = '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/debug_training/images'
    masks_dir = '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/debug_training/masks'
    mean_img = '/home/maryana/storage2/Posdoc/AVID/AT100/slidenet_2classes/training/mean_image.npy'
    img_dim = (204,204,3)
    mask_dim=(200,200)
    nClasses = 2
    batch_size = 32
    do_augmentation = True
    augment_percent = 0.40
    #self,gen_name,images_dir,masks_dir,mean_img,img_dim,mask_dim,nClasses,batch_size,do_augmentation=False,augment_percent=0.40,resize_mask=[]
    tau_gen = TauImageGenerator('Debug',images_dir,masks_dir,mean_img,img_dim,(200,200),nClasses,batch_size,do_augmentation,augment_percent,resize_mask=[200,200],class_weights=[0.8,0.2])

    nEpochs = tau_gen.__len__()
    for i in range(nEpochs):
        #x,y = tau_gen.__getitem__(i)
        x,y,w = tau_gen.get_batch()

if __name__ == "__main__":
    main()