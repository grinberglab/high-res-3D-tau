import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
numbers=[22]

for i in numbers:
    nii1 = nib.load('/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/1stStep_Manual/P2865_+3_22.nii') #load blockface nii
    nii2 = nib.load('/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/1stStep_Manual/mipaved_res10_P2865_+3_22_five_versions_Jonathan_stack_tiles.nii') #load file saved by MIPAV
    nii3 = nib.load('/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/1stStep_Manual/mipaved_res10_P2865_+3_22_five_versions_Jonathan_stack_tiles_mask.nii')
    nii4 = nib.load('/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/1stStep_Manual/mipaved_P2865_+3_22_five_versions_Jonathan_stack_tiles_heatmap_res10.nii')
    img1 = nii1.get_data() #get blockface image matrix

    img2 = nii2.get_data() #get heatmap image matrix
    img3 = nii3.get_data() #get histology image matrix
    img4 = nii4.get_data() #get mask image matrix
    M = nii1.affine #get affine matrix from blockface

    nii5 = nib.Nifti1Image(img2,M) #create new nifti file with histology image and blockface affine matrix
    nii6 = nib.Nifti1Image(img3,M)
    nii7 = nib.Nifti1Image(img4,M)
    nib.save(nii5,'/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/converted_mipaved_res10_P2865_+3_22_five_versions_Jonathan_stack_tiles.nii')
    nib.save(nii6,'/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/converted_mipaved_res10_P2865_+3_22_five_versions_Jonathan_stack_tiles_mask.nii')
    nib.save(nii7,'/home/maryana/R_DRIVE/Experiments/1_End_of_Life/Registration/2.65x_zoom/P2865_+3_22_five_versions_Jonathan_stack_tiles/2ndStep_Automatic/converted_mipaved_P2865_+3_22_five_versions_Jonathan_stack_tiles_heatmap_res10.nii')



