import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from misc.TiffTileLoader import ind2sub,sub2ind
import os

#
# Normalize Batch 1 (Compare with batch 4)
#
def norm_batch2():
    #reference images
    mask_tgt1_path = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_260/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_tgt1_path = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_260/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    mask_ref1_path = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_268/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_ref1_path = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_268/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    av1_batch2 = [116, 124, 132, 140, 236, 244, 252, 260, 356, 364, 372, 380, 476, 484, 492, 500, 596, 604, 612, 620]
    hm_dir = '/home/maryana/storage/Posdoc/AVID/AV23/AT100/full_res/AT100_{}/heat_map/hm_map_0.1/'

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

    for slice in av1_batch2:

        print('Processing slice {}'.format(slice))

        hm_nii_path = os.path.join(hm_dir.format(slice),'heat_map_0.1_res10.nii')
        hm_norm_path = os.path.join(hm_dir.format(slice), 'heat_map_0.1_res10_norm.nii')

        hm_nii = nib.load(hm_nii_path)
        hmap = hm_nii.get_data() / 1000.
        hmap[hmap < 1] = 0

        idx_pixels = np.nonzero(hmap.flatten() > 0)[0]
        nPix = len(idx_pixels)

        # #norm strategy 1
        hmap_norm = np.zeros(hmap.shape)
        for nP in range(nPix):
            idx = idx_pixels[nP]
            (row, col) = ind2sub(hmap.shape, idx)
            hmap_norm[row,col] = hmap[row,col] + bump_val

        norm_nii = nib.Nifti1Image(hmap_norm * 1000., affine=hm_nii.get_affine())
        nib.save(norm_nii,hm_norm_path)


if __name__ == '__main__':
    norm_batch2()