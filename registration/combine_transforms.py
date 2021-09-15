import SimpleITK as sitk


def combine_transforms(tfs_array):

    composite_transform = None
    for tf_pair in tfs_array:

        affine_tf_file = tf_pair[0] #Should be an ANTS affine transform file (XXXX0GenericAffine.mat)
        def_field_file = tf_pair[1] #Should be an Elastix deformation fields (deformationField.nii)

        ants_affine_tf = sitk.ReadTransform(affine_tf_file)
        elastix_def_img = sitk.ReadImage(def_field_file,sitk.sitkVectorFloat64)
        displacement_field_transform = sitk.DisplacementFieldTransform(elastix_def_img)

        if composite_transform == None:
            composite_transform = sitk.Transform(ants_affine_tf)
            composite_transform.AddTransform(displacement_field_transform)
        else:
            composite_transform.AddTransform(ants_affine_tf)
            composite_transform.AddTransform(displacement_field_transform)

    return composite_transform


def get_save_combined_tranform(tfs_array,out_file):
    print('Combining transforms.')

    composite_tf = combine_transforms(tfs_array)
    print('Saving combined transforms to file {}'.format(out_file))
    sitk.WriteTransform(composite_tf, out_file)

    return composite_tf


