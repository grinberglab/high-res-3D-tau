import nibabel as nib
import glob
import numpy as np
import os
import skimage.io as io
from convnet.util.scores import dice_coef2
import pickle
from skimage import img_as_ubyte


#create 3d volume from 2d masks
orig_block_file = '/home/maryana/storage2/Posdoc/AVID/AV23/blockface/nii/av23_blockface.nii'
vent_mask_dir = '/home/maryana/R_DRIVE/Experiments/1_AVID/Cases/1181-002/Master_Package_1181-002/Images/1181-002_BLOCKFACE/dice_masks/ventricles/block'
aparc_file = '/home/maryana/storage2/Posdoc/AVID/AV23/MRI_3D_Registration/AV23_aparc+aseg-to-BlockFace-CorrectHeader-noCerebellumBrainStem-ZeroPadded40_affine1syn.nii'
out_file = '/home/maryana/storage2/Posdoc/AVID/AV23/MRI_3D_Registration/3d_dice_mri_block.pickle'
mri_file = '/home/maryana/storage2/Posdoc/AVID/AV23/MRI_3D_Registration/A13-1181-002-PostMortemSPGR-to-BlockFace-CorrectHeader-noCerebellumBrainStem-ZeroPadded40_affine1syn.nii'
tmp_dir = '/home/maryana/storage2/Posdoc/AVID/AV23/MRI_3D_Registration/mask'


#load parcelation
aparc_nii = nib.load(aparc_file)
aparc = aparc_nii.get_data()

orig_block = nib.load(orig_block_file)
#vol = np.zeros(aparc.shape)

#mri_nii = nib.load(mri_file)
#mri = mri_nii.get_data()

files = glob.glob(os.path.join(vent_mask_dir,'*.tif'))
dices = {}

for file in files:
    block_mask = io.imread(file)
    if block_mask.ndim > 2:
        block_mask = block_mask[...,0]
    block_mask[block_mask > 0] = 255
    block_mask[block_mask != 255] = 0
    #pad mask to match Duygu's padding

    pad1 = np.zeros((20,block_mask.shape[1]))
    block_mask = np.concatenate((pad1,block_mask,pad1),axis=0)
    pad2 = np.zeros((block_mask.shape[0],20))
    block_mask = np.concatenate((pad2,block_mask,pad2),axis=1)

    basename = os.path.basename(file)
    #AV13-002_0207_VM.tif
    str_num = basename[9:13]
    num = int(str_num) #original number
    num += 19 #add shift to match Duygu's padding

    aparc_mask = aparc[...,num]
    mri_mask = np.zeros(aparc_mask.shape)
    mri_mask[aparc_mask == 4] = 255
    mri_mask[aparc_mask == 43] = 255

    D = dice_coef2(block_mask, mri_mask)
    if D < 0.5:
        print('Skipping {}'.format(num-19))
        continue
    dices[basename] = D

    # if D >= 0.7:
    #     io.imsave('{}'.format(num),block_mask)
    # else:
    #     io.imsave('{}'.format(num),img_as_ubyte(mri[:, :, num]))

    #vol[:,:,num] = block_mask

#block_mask_nii = nib.Nifti1Image(vol,aparc_nii.affine)
#nib.save(block_mask_nii,'/home/maryana/storage2/Posdoc/AVID/AV23/MRI_3D_Registration/AV13_block_ventricles_mask_ZeroPadded40.nii')

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













