import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage.measure as meas


ref = sitk.ReadImage('/home/maryana/storage/Posdoc/AVID/AV13/blockface/nii/crop_360.nii')
mov = sitk.ReadImage('/home/maryana/storage/Posdoc/AVID/AV13/AT100/res10/nii/AT100_360_res10.nii')

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetMovingImage(mov)
elastixImageFilter.SetFixedImage(ref)
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap("affine"))
elastixImageFilter.Execute()
sitk.WriteImage(elastixImageFilter.GetResultImage(),'/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res/resize_mary/AT100_360/reg/elastix_AT100_360_affine.nii')

img = elastixImageFilter.GetResultImage()

np_img = sitk.GetArrayFromImage(img)
np_img[np_img < 0] = 0
ref_img = sitk.GetArrayFromImage(ref)
ref_img[ref_img < 0] = 0

img_to_show = np.zeros((np_img.shape[0],np_img.shape[1],3))
img_to_show[:,:,0] = np_img
img_to_show[:,:,1] = sitk.GetArrayFromImage(ref)

plt.imshow(img_to_show)
plt.show()