import os
from combine_transforms import get_save_combined_tranform
import fnmatch
import sys
import SimpleITK as sitk


def get_tform_files(slice_dir):
    tform_dic = {}
    affine_tf_file = ''
    bspline_def_file = ''
    reg_dir = os.path.join(slice_dir,'reg')
    for root, dir, files in os.walk(reg_dir):
        order = -1
        if fnmatch.fnmatch(root, '*/*Step_Auto*'):  # it's inside /RES*
            direc = os.path.split(root)[1]
            order = int(direc[0]) # folder name should always be something like '1stStep:Auto'
            for fn in fnmatch.filter(files, '*0GenericAffine.mat'): #there should be only one
                affine_tf_file = os.path.join(root,fn)
            for fn in fnmatch.filter(files, '*deformationField.nii'): #there should be only one
                bspline_def_file = os.path.join(root,fn)
        if order > -1:
            tform_dic[order] = (affine_tf_file, bspline_def_file)

    return tform_dic


def apply_registrations(slice_dir,mov_file,ref_file,reg_file):

    print('Processing {}'.format(slice_dir))

    tform_dic = get_tform_files(slice_dir)
    key_list = tform_dic.keys()
    key_list.sort()
    nTf = len(key_list)

    print('Found {} transform pair(s)'.format(nTf))

    # create ordered list of transformation pairs
    tform_arr = []
    for k in key_list:
        tform_pair = tform_dic[k]
        tform_arr.append(tform_pair)

    # compute and save combined transform
    combined_tform_file = os.path.join(slice_dir,'reg/combined_transforms.h5')
    composite_tf = get_save_combined_tranform(tform_arr,combined_tform_file)

    # apply transforms
    mov_img = sitk.ReadImage(mov_file, sitk.sitkFloat64)
    ref_img = sitk.ReadImage(ref_file, sitk.sitkFloat64)
    reg_img = sitk.Resample(mov_img, ref_img, composite_tf, sitk.sitkLinear, 0.0, mov_img.GetPixelIDValue())
    sitk.WriteImage(reg_img, reg_file)
    print('Saved registered image.')
    print('Ref image: {}'.format(ref_file))
    print('Mov image: {}'.format(mov_file))
    print('Reg image: {}'.format(reg_file))

def main():
    if len(sys.argv) != 5:
        print('Usage: apply_combined_registrations.py <slice_dir> <mov_file> <ref_file> <reg_file>')
        exit()

    slice_dir = sys.argv[1]
    mov_file = sys.argv[2] #histology or heatmap
    ref_file = sys.argv[3] #blockface
    reg_file = sys.argv[4] #output: registered heatmap/histology

    # slice_dir = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_164'
    # mov_file = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_164/reg/AT100_164_res10.nii'
    # ref_file = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_164/reg/AV13-002_0164.png.nii'
    # reg_file = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_164/reg/reg.nii'

    apply_registrations(slice_dir,mov_file,ref_file,reg_file)

if __name__ == '__main__':
    main()