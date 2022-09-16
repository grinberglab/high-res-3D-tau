import matplotlib as plt
import numpy as np
import nibabel as nib

block_file = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab+4/blockface/seg/small/slab+4.nii'
block_file2 = '/home/maryana/storage/Posdoc/EndOfLife/P2724/slab+4/blockface/seg/small/slab+4_2.nii'
mri_file = '/home/maryana/storage/Posdoc/EndOfLife/P2724/mri/mri_brain_0.5.nii.gz'

block = nib.load(block_file)
data_b = block.get_data()
print(data_b.shape)

mri = nib.load(mri_file)
data_m = mri.get_data()
print(data_m.shape)

affine_m = mri.affine
print(affine_m)

affine_b = block.affine
print(affine_b)

block2 = nib.Nifti1Image(data_b, affine_m)
nib.save(block2, block_file2)