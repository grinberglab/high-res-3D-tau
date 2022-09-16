import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib as mpl
import matplotlib.cm as cm
from skimage import img_as_ubyte

nii = nib.load('/home/maryana/storage2/Posdoc/AVID/AV23/blockface/nii/av23_blockface.nii')

#nii = nib.load('/homecreate_heatmap2block.py/maryana/storage2/Posdoc/AVID/AV23/blockface/nii/av23_blockface.nii')
vol = nii.get_data()
vol2 = np.zeros(vol.shape)


slices = [76,84,92,98,100,116,124,132,140,148,156,164,172,180,188,196,204,212,220,228,236,244,252,260,
          268,276,284,292,300,308,316,324,332,340,348,356,364,372,380,388,396,404,412,420,428,436,444,452,
          460,468,476,484,492,500,508,516,524,532,540,548,556,564,572,580,588,596,604,612,620,628,636,644,
          652,660,668,673,681,689,700,705,713,721,732,737,745,753,764,762] #last element is a pivot and wont be included

slice_name='/home/maryana/R_DRIVE/Experiments/1_AVID/Cases/1181-002/Master_Package_1181-002' \
           '/Images/AT100/full_res/AT100_{}/reg/combined_AT100_{}_heatmap_113018.nii'
#slice_name='/home/maryana/storage2/Posdoc/AVID/AV13/AT100/full_res/AT100_{}/reg/combined_AT100_{}_01072019.nii'
#slice_name = '/home/maryana/storage2/Posdoc/AVID/AV23/AT100/registered_files/combined_AT100_{}_heatmap_113018.nii'
#slice_name = '/home/maryana/R_DRIVE/Experiments/1_AVID/Cases/1811-001/Master_Package_1181-001/Images/1181-001_registratrion/AV1_AT8_Registrations/Completed/{}/reg/combined_AT8_{}_heatmap_010419.nii'


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
hm_name='/home/maryana/storage2/Posdoc/AVID/AV13/blockface/nii/AT8_heatmap2blockface_amira_test.nii'
nii2 = nib.Nifti1Image(vol2,nii.affine)
nib.save(nii2,hm_name)
print('Files {} sucessfully saved.'.format(hm_name))