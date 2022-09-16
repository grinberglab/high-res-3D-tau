import nibabel as nib
import glob
import numpy as np
import os
import skimage.io as io
from convnet.util.scores import dice_coef2
import pickle


#create 3d volume from 2d masks
orig_block_file = '/home/maryana/storage2/Posdoc/AVID/AV13/blockface/nii/av13_blockface.nii'
vent_mask_dir = '/home/maryana/R_DRIVE/Experiments/1_AVID/Cases/1811-001/Master_Package_1181-001/Images/1181-001_BLOCKFACE/dice_masks/ventricles/block/ventricle_masks'
aparc_file = '/home/maryana/storage2/Posdoc/AVID/AV13/MRI_3D_Registration/AV13_aparc+aseg-to-BlockFace-CorrectHeader-CropLeft200_affine1syn.nii'
out_file = '/home/maryana/storage2/Posdoc/AVID/AV13/MRI_3D_Registration/3d_dice_mri_block.pickle'


#load parcelation
aparc_nii = nib.load(aparc_file)
aparc = aparc_nii.get_data()


orig_block = nib.load(orig_block_file)


files = glob.glob(os.path.join(vent_mask_dir,'*.tif'))
dices = {}
for file in files:
    block_mask = io.imread(file)
    if block_mask.ndim > 2:
        block_mask = block_mask[...,0]
    block_mask[block_mask > 0] = 255
    block_mask[block_mask != 255] = 0

    #crop volume to match Duygu's volumes
    block_mask = block_mask[:,0:-200]

    basename = os.path.basename(file)
    #1181_001-Whole-Brain_0366.tif
    str_num = basename[21:25]
    num = int(str_num)

    aparc_mask = aparc[...,num]
    mri_mask = np.zeros(aparc_mask.shape)
    mri_mask[aparc_mask == 4] = 255
    mri_mask[aparc_mask == 43] = 255

    D = dice_coef2(block_mask, mri_mask)
    dices[basename] = D



pickle_out = open(out_file,'wb')
pickle.dump(dices,pickle_out)
pickle_out.close()


nDices = len(dices.keys())
dices_arr = np.zeros((nDices))
count = 0
for k in dices.keys():
    dices_arr[count] = dices[k]
    count += 1
uDice = np.mean(dices_arr)
stdDice = np.std(dices_arr)
print('Mean Dice: {} | Std: {}'.format(uDice,stdDice))













