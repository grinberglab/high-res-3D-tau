import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from misc.TiffTileLoader import ind2sub,sub2ind
import os

#
# Normalize Batch 8 (Compare with batch 7)
#
def norm_batch8():
    #reference images
    mask_tgt1_path = '/home/maryana/storage2/Posdoc/AVID/AV23/AT8/full_res/AT8_511/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_tgt1_path = '/home/maryana/storage2/Posdoc/AVID/AV23/AT8/full_res/AT8_511/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    mask_ref1_path = '/home/maryana/storage2/Posdoc/AVID/AV23/AT8/full_res/AT8_503/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_ref1_path = '/home/maryana/storage2/Posdoc/AVID/AV23/AT8/full_res/AT8_503/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    av1_batch8 = [347]
    hm_dir = '/home/maryana/R_DRIVE/Experiments/1_AVID/graphs_for_paper/'

    #load target and ref 1
    mask_nii_ref1 = nib.load(mask_ref1_path)
    hm_nii_ref1 = nib.load(hm_ref1_path)
    mask_ref1 = mask_nii_ref1.get_data()
    hm_ref1 = hm_nii_ref1.get_data()
    mask_ref1 = mask_ref1.squeeze(axis=2).astype('uint8')
    pixels_ref1 = hm_ref1[mask_ref1 == 1]
    pixels_ref1 /= 1000.

    mask_nii_tgt1 = nib.load(mask_tgt1_path)
    hm_nii_tgt1 = nib.load(hm_tgt1_path)
    mask_tgt1 = mask_nii_tgt1.get_data()
    hm_tgt1 = hm_nii_tgt1.get_data()
    mask_tgt1 = mask_tgt1.squeeze(axis=2).astype('uint8')
    pixels_tgt1 = hm_tgt1[mask_tgt1 == 1]
    pixels_tgt1 /= 1000.

    q1_tgt1, q3_tgt1 = np.percentile(pixels_tgt1.flatten(), 25),np.percentile(pixels_tgt1.flatten(), 75)  #target
    q1_ref1, q3_ref1 = np.percentile(pixels_ref1.flatten(), 25),np.percentile(pixels_ref1.flatten(), 75)  #reference

    cutoff_tgt1 = (q3_tgt1 - q1_tgt1) * 1.5
    upper_tgt1 = q3_tgt1 + cutoff_tgt1
    cutoff_ref1 = (q3_ref1 - q1_ref1) * 1.5
    upper_ref1 = q3_ref1 + cutoff_ref1

    pixel_ref1 = pixels_ref1[pixels_ref1 < upper_ref1]
    pixel_tgt1 = pixels_tgt1[pixels_tgt1 < upper_tgt1]

    mean_pixel_ref = pixel_ref1.mean()
    mean_pixel_tgt = pixel_tgt1.mean()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([pixels_ref1.flatten(),pixels_tgt1.flatten()], notch = True,  labels=['Ref1','Tgt1'])

    bump_val = np.abs(mean_pixel_ref - mean_pixel_tgt)

    for slice in av1_batch8:

        try:
            print('Processing slice {}'.format(slice))

            hm_nii_path = os.path.join(hm_dir.format(slice),'A13-1181-001-MC1-CorrectHeader-CropLeft200-double_mask_for_paper_2d.nii')
            hm_norm_path = os.path.join(hm_dir.format(slice), 'A13-1181-001-MC1-CorrectHeader-CropLeft200-double_mask_for_paper_2d_norm_0_8.nii')

            hm_nii = nib.load(hm_nii_path)
            hmap = hm_nii.get_data() / 1000.
            #hmap[hmap < 1] = 0

            idx_pixels = np.nonzero(hmap.flatten() > 0)[0]
            nPix = len(idx_pixels)

            # #norm strategy 1
            hmap_norm = np.zeros(hmap.shape)
            for nP in range(nPix):
                idx = idx_pixels[nP]
                (row, col) = ind2sub(hmap.shape, idx)
                hmap_norm[row,col] = hmap[row,col] + bump_val * 0.8

            norm_nii = nib.Nifti1Image(hmap_norm * 1000., affine=hm_nii.get_affine())
            nib.save(norm_nii,hm_norm_path)



        except:
            print('Could not process slice {}'.format(slice))

        # for nature_paper
        hm_nii_path_2 = os.path.join(hm_dir.format(slice), 'A13-1181-001-MC1-CorrectHeader-CropLeft200-double_mask_for_paper_2d_norm_0_8.nii')
        hm_norm_path_2 = os.path.join(hm_dir.format(slice), 'A13-1181-001-MC1-CorrectHeader-CropLeft200-double_mask_for_paper_2d_norm_0_8_1_5_22000.nii')
        hm_nii_2 = nib.load(hm_nii_path_2)
        hmap_2 = hm_nii_2.get_data()
        idx_pixels_2 = np.nonzero(hmap_2.flatten() > 22000)[0]
        nPix_2 = len(idx_pixels_2)

        for nP_2 in range(nPix_2):
            idx_2 = idx_pixels_2[nP_2]
            (row, col) = ind2sub(hmap_2.shape, idx_2)
            hmap_2[row, col] = hmap_2[row, col] * 1.5

        norm_nii_2 = nib.Nifti1Image(hmap_2, affine=hm_nii_2.get_affine())
        nib.save(norm_nii_2, hm_norm_path_2)


if __name__ == '__main__':
    norm_batch8()