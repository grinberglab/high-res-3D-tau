import os
from combine_transforms import get_save_combined_tranform
import fnmatch
import sys
import SimpleITK as sitk

def apply_registration(mov_file,ref_file,reg_file,tform_file):
    mov_img = sitk.ReadImage(mov_file, sitk.sitkFloat64)
    ref_img = sitk.ReadImage(ref_file, sitk.sitkFloat64)
    tform = sitk.ReadTransform(tform_file)
    reg_img = sitk.Resample(mov_img, ref_img, tform, sitk.sitkLinear, 0.0, mov_img.GetPixelIDValue())

    print('Writing {}'.format(reg_file))
    sitk.WriteImage(reg_img, reg_file)

def main():
    if len(sys.argv) != 5:
        print('Usage: apply_precombined_registrations.py <mov_file> <ref_file> <reg_file> <tform_file>')
        exit()

    mov_file = sys.argv[1] #histology or heatmap
    ref_file = sys.argv[2] #blockface
    reg_file = sys.argv[3] #output: registered heatmap/histology    tform_file = sys.argv[4]
    tform_file = sys.argv[4]

    apply_registration(mov_file,ref_file,reg_file,tform_file)

if __name__ == '__main__':
    main()