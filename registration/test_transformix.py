import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage.measure as meas


slice_id='360'
reg_dir='/home/maryana/storage/Posdoc/AVID/AV13/AT100/full_res/resize_mary/AT100_'+slice_id+'/reg/'
block_img='1181_001-Whole-Brain_0'+slice_id+'.png.nii'

ref = sitk.ReadImage('/home/maryana/storage/Posdoc/AVID/AV13/blockface/nii/'+block_img)
mov = sitk.ReadImage(reg_dir+'ants_AT100_'+slice_id+'_heatmap_affine.nii')

transformixImageFilter = sitk.TransformixImageFilter()
transformixImageFilter.SetOutputDirectory(reg_dir)
transformixImageFilter.SetMovingImage(mov)

tform1 = sitk.ReadParameterFile(reg_dir+'TransformParameters.0.txt')
tform2 = sitk.ReadParameterFile(reg_dir+'TransformParameters.1.txt')
tform3 = sitk.ReadParameterFile(reg_dir+'TransformParameters.2.txt')
tform4 = sitk.ReadParameterFile(reg_dir+'TransformParameters.3.txt')
tform5 = sitk.ReadParameterFile(reg_dir+'TransformParameters.4.txt')
tform6 = sitk.ReadParameterFile(reg_dir+'TransformParameters.5.txt')

transformixImageFilter.AddTransformParameterMap(tform1)
transformixImageFilter.AddTransformParameterMap(tform2)
transformixImageFilter.AddTransformParameterMap(tform3)
transformixImageFilter.AddTransformParameterMap(tform4)
transformixImageFilter.AddTransformParameterMap(tform5)
transformixImageFilter.AddTransformParameterMap(tform6)

transformixImageFilter.LogToConsoleOn()

transformixImageFilter.Execute()

warp = transformixImageFilter.GetComputeDeformationField()


out_img_file = reg_dir+'elastix_'+slice_id+'_heatmap_bspline_se.nii'
sitk.WriteImage(transformixImageFilter.GetResultImage(),out_img_file)


img = transformixImageFilter.GetResultImage()

np_img = sitk.GetArrayFromImage(img)
np_img[np_img < 0] = 0
ref_img = sitk.GetArrayFromImage(ref)
ref_img[ref_img < 0] = 0

img_to_show = np.zeros((np_img.shape[0],np_img.shape[1],3))
img_to_show[:,:,0] = np_img
img_to_show[:,:,1] = sitk.GetArrayFromImage(ref)

plt.imshow(img_to_show)
plt.show()

print(out_img_file)
