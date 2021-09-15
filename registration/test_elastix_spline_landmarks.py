import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage.measure as meas



ref = sitk.ReadImage('/home/maryana/storage/Posdoc/AVID/AV13/blockface/nii/1181_001-Whole-Brain_0360.png.nii')
mov = sitk.ReadImage('/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res/resize_mary/AT100_360/reg/ants_AT100_360_affine.nii')

parameterMap = sitk.GetDefaultParameterMap("bspline")
parameterMap['Transform'] = ['SplineKernelTransform']
#parameterMap['SplineKernelType'] = ['ThinPlateSpline']

#parameterMapVector = sitk.VectorOfParameterMap()
# parameterMapVector.append(parameterMap)

elastixImageFilter = sitk.ElastixImageFilter()

elastixImageFilter.SetParameterMap(parameterMap)
elastixImageFilter.LogToConsoleOn()
elastixImageFilter.PrintParameterMap()

elastixImageFilter.SetFixedImage(ref)
elastixImageFilter.SetMovingImage(mov)
elastixImageFilter.SetFixedPointSetFileName('/home/maryana/storage/Posdoc/AVID/AV13/test_elastix/ref.pts')
#elastixImageFilter.SetMovingPointSetFileName('/home/maryana/storage/Posdoc/AVID/AV13/test_elastix/mov_ants_ctr.pts')

#sitk.WriteParameterFile(elastixImageFilter.GetParameterMap()[0], '/home/maryana/storage/Posdoc/AVID/AV13/test_elastix/params.txt')

elastixImageFilter.Execute()



sitk.WriteImage(elastixImageFilter.GetResultImage(),'/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res/resize_mary/AT100_360/output/RES(10x10)/reg/elastix_AT100_.nii')
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
pass