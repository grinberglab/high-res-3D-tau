import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib as mpl
import matplotlib.cm as cm
from skimage import img_as_ubyte

nii = nib.load('/home/maryana/storage2/Posdoc/AVID/AV13/blockface/nii/av13_blockface.nii')

#nii = nib.load('/homecreate_heatmap2block.py/maryana/storage2/Posdoc/AVID/AV23/blockface/nii/av23_blockface.nii')
vol = nii.get_data()
vol2 = np.zeros(vol.shape)


slices = [280,296,312,328,344,360,376,392,408,424,440,457,472,488,504,520,536,552,568,584,600,616,632,648,654] #last element is a pivot and wont be included

slice_name='/home/maryana/storage2/Posdoc/AVID/AV13/AT100/full_res/AT100_{}/reg/combined_AT100_{}_heatmap_102518.nii'


nSlices = len(slices)
for i in range(0,nSlices):

    if i == nSlices-1:
        break #dont process pivot element

    id = slices[i]
    id-=1
    id2 = slices[i+1]
    id2-=1
    try:
        name = slice_name.format(str(id+1),str(id+1))
        slice = nib.load(name)
        simg = slice.get_data()
        for s in range(id,id2):
            vol2[:,:,s] = simg

    except Exception as e:
        print("Error loading slice {}".format(id+1))
        print(e)


#hm_name='/home/maryana/storage2/Posdoc/AVID/AV23/blockface/nii/AT100_heatmap2blockface_100818_QC_OK.nii'
hm_name='/home/maryana/storage2/Posdoc/AVID/AV13/blockface/nii/AT100_heatmap2blockface_amira_test.nii'
nii2 = nib.Nifti1Image(vol2,nii.affine)
nib.save(nii2,hm_name)
print('Files {} sucessfully saved.'.format(hm_name))