import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys
import skimage.measure as meas


slice_id='322'

block_img='1181_001-Whole-Brain_0'+slice_id+'.png.nii'
#block_img='crop_'+slice_id+'.nii'

reg_dir = '/home/maryana/storage/Posdoc/AVID/AV13/AT8/full_res/slices/AT8_{}/reg/'.format(slice_id)

ref = sitk.ReadImage('/home/maryana/storage/Posdoc/AVID/AV13/blockface/nii/'+block_img)
mov = sitk.ReadImage(reg_dir+'ants_AT8_'+slice_id+'_affine.nii')
#ref_mask = sitk.ReadImage(folder+'MASK_1181_001-Whole-Brain_0457_v2.png.nii',sitk.sitkUInt8)


parameterMap = sitk.GetDefaultParameterMap("bspline")
del parameterMap['FinalGridSpacingInPhysicalUnits']

# elastixImageFilter = sitk.ElastixImageFilter()
# elastixImageFilter.SetMovingImage(mov)
# elastixImageFilter.SetFixedImage(ref)
# elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap("affine"))
# elastixImageFilter.Execute()
# sitk.WriteImage(elastixImageFilter.GetResultImage(),'/home/maryana/storage/Posdoc/AVID/test_elastix/elastix_AT100_457_h2b_noland.nii')

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetOutputDirectory(reg_dir)
# elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('translation'))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('rigid'))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
#elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
# elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('bspline'))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('bspline'))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('bspline'))
# elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('bspline'))
#elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('bspline'))

elastixImageFilter.SetParameterMap(parameterMap)
elastixImageFilter.AddParameterMap(parameterMap)
elastixImageFilter.AddParameterMap(parameterMap)
elastixImageFilter.AddParameterMap(parameterMap)
elastixImageFilter.AddParameterMap(parameterMap)
elastixImageFilter.AddParameterMap(parameterMap)
#elastixImageFilter.AddParameterMap(parameterMap)

elastixImageFilter.SetParameter('FinalBSplineInterpolationOrder','0')


elastixImageFilter.SetParameter(0,'FinalGridSpacingInVoxels','100')
elastixImageFilter.SetParameter(1,'FinalGridSpacingInVoxels','80')
elastixImageFilter.SetParameter(2,'FinalGridSpacingInVoxels','50')
elastixImageFilter.SetParameter(3,'FinalGridSpacingInVoxels','30')
elastixImageFilter.SetParameter(4,'FinalGridSpacingInVoxels','20')
elastixImageFilter.SetParameter(5,'FinalGridSpacingInVoxels','10')
#elastixImageFilter.SetParameter(6,'FinalGridSpacingInVoxels','5')


#elastixImageFilter.SetParameter(4,'FinalGridSpacingInPhysicalUnits','2.0')



elastixImageFilter.SetMovingImage(mov)
elastixImageFilter.SetFixedImage(ref)
#elastixImageFilter.SetFixedMask(ref_mask)


#elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap('affine'))

elastixImageFilter.LogToConsoleOn()
#elastixImageFilter.SetParameter("Registration","MultiMetricMultiResolutionRegistration")
# ##elastixImageFilter.SetParameter( "Metric", ("NormalizedMutualInformation", "CorrespondingPointsEuclideanDistanceMetric",))
# #elastixImageFilter.SetParameter(1,"Metric0Weight", "0.0")
# #elastixImageFilter.SetParameter(1,"Metric1Weight", "10.0")
#elastixImageFilter.AddParameter("Metric", "CorrespondingPointsEuclideanDistanceMetric" )
# elastixImageFilter.SetParameter("Metric0Weight", "0")
# elastixImageFilter.SetParameter("Metric1Weight", "0")
# elastixImageFilter.SetParameter("Metric2Weight", "1.0")
#
# elastixImageFilter.SetFixedPointSetFileName("/home/maryana/storage/Posdoc/AVID/test_elastix/ref_ants_ctr.pts")
# elastixImageFilter.SetMovingPointSetFileName("/home/maryana/storage/Posdoc/AVID/test_elastix/mov_ants_ctr.pts")

# elastixImageFilter.SetParameter("MaximumNumberOfIterations" , str(iterationNumbers))
# elastixImageFilter.SetParameter("NumberOfSpatialSamples" , str(spatialSamples))

elastixImageFilter.PrintParameterMap()

elastixImageFilter.Execute()
out_img_file = reg_dir+'/elastix_AT8_'+slice_id+'_bspline.nii'
sitk.WriteImage(elastixImageFilter.GetResultImage(),out_img_file)


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

print(out_img_file)