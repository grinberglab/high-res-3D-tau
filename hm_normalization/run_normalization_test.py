import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from misc.TiffTileLoader import ind2sub,sub2ind
import os

def rescale(x,tgt_a,tgt_b,ref_a,ref_b):
    y = (((ref_b - ref_a) * (x - tgt_a))/tgt_b - tgt_a) + ref_a
    return y

#
# Normalize Batch 1 (Compare with batch 4)
#
def norm_batch1():
    #reference images
    mask418_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_418/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii.gz'
    img418_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_418/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'
    mask422_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_422/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii.gz'
    img422_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_422/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'
    mask434_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_434/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii.gz'
    img434_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_434/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'
    mask438_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_438/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii.gz'
    img438_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_438/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    to_fix_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_410/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'
    norm_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_410/heat_map/hm_map_0.1/heat_map_0.1_res10_norm.nii'

    av1_batch1 = [282, 290, 298, 306, 314, 322, 330, 338, 346, 354, 362, 370, 378, 386, 394, 402, 410, 418, 426, 434]
    hm_dir = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_{}/heat_map/hm_map_0.1/'


    mask_nii_418 = nib.load(mask418_path)
    hm_nii_418 = nib.load(img418_path)
    mask_418 = mask_nii_418.get_data()
    hm_418 = hm_nii_418.get_data()
    mask_418 = mask_418.squeeze(axis=2).astype('uint8')
    pixels_418 = hm_418[mask_418 == 1]
    pixels_418 /= 1000.


    mask_nii_422 = nib.load(mask422_path)
    hm_nii_422 = nib.load(img422_path)
    mask_422 = mask_nii_422.get_data()
    hm_422 = hm_nii_422.get_data()
    mask_422 = mask_422.squeeze(axis=2).astype('uint8')
    pixels_422 = hm_422[mask_422 == 1]
    pixels_422 /= 1000.

    # test_img = np.zeros(hm_422.shape)
    # idx_pixels = np.nonzero(mask_422.flatten() == 1)[0]
    # nPix = len(idx_pixels)
    # for nP in range(nPix):
    #     idx = idx_pixels[nP]
    #     (row,col) = ind2sub(hm_422.shape,idx)
    #     test_img[row,col] = hm_422[row,col] / 1000.
    #
    # test_img2 = test_img.copy()



    mask_nii_434 = nib.load(mask434_path)
    hm_nii_434 = nib.load(img434_path)
    mask_434 = mask_nii_434.get_data()
    hm_434 = hm_nii_434.get_data()
    mask_434 = mask_434.squeeze(axis=2).astype('uint8')
    pixels_434 = hm_434[mask_434 == 1]
    pixels_434 /= 1000.

    mask_nii_438 = nib.load(mask438_path)
    hm_nii_438 = nib.load(img438_path)
    mask_438 = mask_nii_438.get_data()
    hm_438 = hm_nii_438.get_data()
    mask_438 = mask_438.squeeze(axis=2).astype('uint8')
    pixels_438 = hm_438[mask_438 == 1]
    pixels_438 /= 1000.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([pixels_418.flatten(),pixels_422.flatten()], notch = True,  labels=['418','422'])

    q1_418, q3_418 = np.percentile(pixels_418.flatten(), 25),np.percentile(pixels_418.flatten(), 75)  #target
    q1_422, q3_422 = np.percentile(pixels_422.flatten(), 25),np.percentile(pixels_422.flatten(), 75)  #reference
    #bump_val = np.abs(q3_418 - q3_422)

    cutoff_tgt = (q3_418 - q1_418) * 1.5
    upper_tgt = q3_418 + cutoff_tgt

    cutoff_ref = (q3_422 - q1_422) * 1.5
    upper_ref = q3_422 + cutoff_ref

    pixel_ref = pixels_422[pixels_422 < upper_ref]
    pixel_tgt = pixels_418[pixels_418 < upper_tgt]
    bump_val = np.abs(pixel_ref.mean() - pixel_tgt.mean())

    # test_img2[test_img2 > q3_422] = 0

    for slice in av1_batch1:

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

        # # #norm strategy 2
        # norm_img2 = np.zeros(img.shape)
        # for nP in range(nPix):
        #     idx = idx_pixels[nP]
        #     (row, col) = ind2sub(img.shape, idx)
        #     x = img[row,col]
        #     y = rescale(x,0,q3_418,0,q3_422)
        #     norm_img2[row,col] = y

        #131 means 1x3 grid, subplot 1
        # fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax1.imshow(hmap)
        # ax2 = fig.add_subplot(122)
        # ax2.imshow(hmap_norm)
        #
        # pixels_img = hmap[hmap > 0]
        # pixels_norm = hmap_norm[hmap_norm > 0]


        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.boxplot([pixels_img.flatten(),pixels_norm.flatten()], notch = True,  labels=['Orig img','Norm img 1'])

        norm_nii = nib.Nifti1Image(hmap_norm * 1000., affine=hm_nii.get_affine())
        nib.save(norm_nii,hm_norm_path)


#
# Normalize batch 2 (Compare with batch 4)
#

def print_batch_graphs():
    av1_batch1 = [282, 290, 298, 306, 314, 322, 330, 338, 346, 354, 362, 370, 378, 386, 394, 402, 410, 418, 426, 434]
    av1_batch2 = [442, 450, 458, 466, 474, 482, 490, 498, 506, 514, 522, 530, 538, 546, 554, 562, 570, 578, 586, 594]
    av1_batch3 = [286, 294, 302, 310, 318, 326, 334, 342, 350, 358, 366, 602, 610, 618, 626, 634, 642, 650, 658, 666]
    av1_batch4 = [374, 382, 390, 398, 406, 414, 422, 430, 438, 448, 454, 462, 470, 478, 486, 494, 502, 510, 518, 526, 534, 542, 550, 558, 566, 574, 582, 590, 598, 606]
    hm_dir = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_{}/heat_map/hm_map_0.1/'

    pixels_b1 = []
    pixels_b2 = []
    pixels_b3 = []
    pixels_b4 = []


    for slice in av1_batch1:
        try:
            print('Processing slice {}'.format(slice))

            hm_nii_path = os.path.join(hm_dir.format(slice),'heat_map_0.1_res10.nii')
            hm_nii = nib.load(hm_nii_path)
            hmap = hm_nii.get_data() / 1000.
            hmap[hmap < 1] = 0

            if pixels_b1 == []:
                pixels_b1 = hmap[hmap > 0]
                pixels_b1 = pixels_b1.flatten()
            else:
                tmp = hmap[hmap > 0]
                pixels_b1 = np.concatenate((pixels_b1,tmp.flatten()),axis=0)
        except:
            print('Batch 1: Error')


    for slice in av1_batch2:

        try:
            print('Processing slice {}'.format(slice))

            hm_nii_path = os.path.join(hm_dir.format(slice),'heat_map_0.1_res10.nii')
            hm_nii = nib.load(hm_nii_path)
            hmap = hm_nii.get_data() / 1000.
            hmap[hmap < 1] = 0

            if pixels_b2 == []:
                pixels_b2 = hmap[hmap > 0]
                pixels_b2 = pixels_b2.flatten()
            else:
                tmp = hmap[hmap > 0]
                pixels_b2 = np.concatenate((pixels_b2,tmp.flatten()),axis=0)
        except:
            print('Batch 2: Error')


    for slice in av1_batch3:

        try:
            print('Processing slice {}'.format(slice))

            hm_nii_path = os.path.join(hm_dir.format(slice),'heat_map_0.1_res10.nii')
            hm_nii = nib.load(hm_nii_path)
            hmap = hm_nii.get_data() / 1000.
            hmap[hmap < 1] = 0

            if pixels_b3 == []:
                pixels_b3 = hmap[hmap > 0]
                pixels_b3 = pixels_b3.flatten()
            else:
                tmp = hmap[hmap > 0]
                pixels_b3 = np.concatenate((pixels_b3,tmp.flatten()),axis=0)
        except:
            print('Batch3: Error')

    for slice in av1_batch4:

        try:
            print('Processing slice {}'.format(slice))

            hm_nii_path = os.path.join(hm_dir.format(slice),'heat_map_0.1_res10.nii')
            hm_nii = nib.load(hm_nii_path)
            hmap = hm_nii.get_data() / 1000.
            hmap[hmap < 1] = 0

            if pixels_b4 == []:
                pixels_b4 = hmap[hmap > 0]
                pixels_b4 = pixels_b4.flatten()
            else:
                tmp = hmap[hmap > 0]
                pixels_b4 = np.concatenate((pixels_b4,tmp.flatten()),axis=0)
        except:
            print('Batch 4: error')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([pixels_b1], notch = True,  labels=['Batch1'])




def norm_batch4():
    #reference images

    #Target: Batch 4
    mask_tgt1_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_462/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_tgt1_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_462/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'
    mask_tgt2_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_470/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_tgt2_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_470/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    #Reference: Batch 2
    mask_ref1_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_466/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_ref1_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_466/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'
    mask_ref2_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_474/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_ref2_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_474/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    av1_batch4 = [374,414,454,494,534,574,382,422,462,502,542,582,390,430,470,510,550,590,398,438,478,518,558,598,406,448,486,526,566,606]
    hm_dir = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_{}/heat_map/hm_map_0.1/'

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

    #load target and ref 2
    mask_nii_ref2 = nib.load(mask_ref2_path)
    hm_nii_ref2 = nib.load(hm_ref2_path)
    mask_ref2 = mask_nii_ref2.get_data()
    hm_ref2 = hm_nii_ref2.get_data()
    mask_ref2 = mask_ref2.squeeze(axis=2).astype('uint8')
    pixels_ref2 = hm_ref2[mask_ref2 == 1]
    pixels_ref2 /= 1000.

    mask_nii_tgt2 = nib.load(mask_tgt2_path)
    hm_nii_tgt2 = nib.load(hm_tgt2_path)
    mask_tgt2 = mask_nii_tgt2.get_data()
    hm_tgt2 = hm_nii_tgt2.get_data()
    mask_tgt2 = mask_tgt2.squeeze(axis=2).astype('uint8')
    pixels_tgt2 = hm_tgt2[mask_tgt2 == 1]
    pixels_tgt2 /= 1000.

    q1_tgt1, q3_tgt1 = np.percentile(pixels_tgt1.flatten(), 25),np.percentile(pixels_tgt1.flatten(), 75)  #target
    q1_ref1, q3_ref1 = np.percentile(pixels_ref1.flatten(), 25),np.percentile(pixels_ref1.flatten(), 75)  #reference

    q1_tgt2, q3_tgt2 = np.percentile(pixels_tgt2.flatten(), 25),np.percentile(pixels_tgt2.flatten(), 75)  #target
    q1_ref2, q3_ref2 = np.percentile(pixels_ref2.flatten(), 25),np.percentile(pixels_ref2.flatten(), 75)  #reference

    cutoff_tgt1 = (q3_tgt1 - q1_tgt1) * 1.5
    upper_tgt1 = q3_tgt1 + cutoff_tgt1
    cutoff_ref1 = (q3_ref1 - q1_ref1) * 1.5
    upper_ref1 = q3_ref1 + cutoff_ref1

    cutoff_tgt2 = (q3_tgt2 - q1_tgt2) * 1.5
    upper_tgt2 = q3_tgt2 + cutoff_tgt2
    cutoff_ref2 = (q3_ref2 - q1_ref2) * 1.5
    upper_ref2 = q3_ref2 + cutoff_ref2

    pixel_ref1 = pixels_ref1[pixels_ref1 < upper_ref1]
    pixel_tgt1 = pixels_tgt1[pixels_tgt1 < upper_tgt1]

    pixel_ref2 = pixels_ref2[pixels_ref2 < upper_ref2]
    pixel_tgt2 = pixels_tgt2[pixels_tgt2 < upper_tgt2]

    mean_pixel_ref = (pixel_ref1.mean() + pixel_ref2.mean())/2
    mean_pixel_tgt = (pixel_tgt1.mean() + pixel_tgt2.mean())/2

    bump_val = np.abs(mean_pixel_ref - mean_pixel_tgt)

    for slice in av1_batch4:

        try:

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

        except Exception as ex:
            print(ex)



#
# Normalize batch 3 (Compare with batch 1 after normalization)
#
def norm_batch3():
    #reference images
    mask_ref1_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_330/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_ref1_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_330/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'
    mask_ref2_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_306/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_ref2_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_306/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    mask_tgt1_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_326/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_tgt1_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_326/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'
    mask_tgt2_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_310/heat_map/hm_map_0.1/heat_map_0.1_res10_mask4norm.nii'
    hm_tgt2_path = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_310/heat_map/hm_map_0.1/heat_map_0.1_res10.nii'

    av1_batch3 = [286, 294, 302, 310, 318, 326, 334, 342, 350, 358, 366, 602, 610, 618, 626, 634, 642, 650, 658, 666]
    hm_dir = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/full_res/AT8_{}/heat_map/hm_map_0.1/'

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

    #load target and ref 2
    mask_nii_ref2 = nib.load(mask_ref2_path)
    hm_nii_ref2 = nib.load(hm_ref2_path)
    mask_ref2 = mask_nii_ref2.get_data()
    hm_ref2 = hm_nii_ref2.get_data()
    mask_ref2 = mask_ref2.squeeze(axis=2).astype('uint8')
    pixels_ref2 = hm_ref2[mask_ref2 == 1]
    pixels_ref2 /= 1000.

    mask_nii_tgt2 = nib.load(mask_tgt2_path)
    hm_nii_tgt2 = nib.load(hm_tgt2_path)
    mask_tgt2 = mask_nii_tgt2.get_data()
    hm_tgt2 = hm_nii_tgt2.get_data()
    mask_tgt2 = mask_tgt2.squeeze(axis=2).astype('uint8')
    pixels_tgt2 = hm_tgt2[mask_tgt2 == 1]
    pixels_tgt2 /= 1000.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([pixels_ref1.flatten(),pixels_tgt1.flatten(),pixels_ref2.flatten(),pixels_tgt2.flatten()], notch = True,  labels=['Ref1','Tgt1', 'Ref2', 'Tgt2'])

    q1_tgt1, q3_tgt1 = np.percentile(pixels_tgt1.flatten(), 25),np.percentile(pixels_tgt1.flatten(), 75)  #target
    q1_ref1, q3_ref1 = np.percentile(pixels_ref1.flatten(), 25),np.percentile(pixels_ref1.flatten(), 75)  #reference

    q1_tgt2, q3_tgt2 = np.percentile(pixels_tgt2.flatten(), 25),np.percentile(pixels_tgt2.flatten(), 75)  #target
    q1_ref2, q3_ref2 = np.percentile(pixels_ref2.flatten(), 25),np.percentile(pixels_ref2.flatten(), 75)  #reference

    cutoff_tgt1 = (q3_tgt1 - q1_tgt1) * 1.5
    upper_tgt1 = q3_tgt1 + cutoff_tgt1
    cutoff_ref1 = (q3_ref1 - q1_ref1) * 1.5
    upper_ref1 = q3_ref1 + cutoff_ref1

    cutoff_tgt2 = (q3_tgt2 - q1_tgt2) * 1.5
    upper_tgt2 = q3_tgt2 + cutoff_tgt2
    cutoff_ref2 = (q3_ref2 - q1_ref2) * 1.5
    upper_ref2 = q3_ref2 + cutoff_ref2

    pixel_ref1 = pixels_ref1[pixels_ref1 < upper_ref1]
    pixel_tgt1 = pixels_tgt1[pixels_tgt1 < upper_tgt1]

    pixel_ref2 = pixels_ref2[pixels_ref2 < upper_ref2]
    pixel_tgt2 = pixels_tgt2[pixels_tgt2 < upper_tgt2]

    #mean_pixel_ref = (pixel_ref1.mean() + pixels_ref2.mean())/2
    #mean_pixel_tgt = (pixel_tgt1.mean() + pixel_tgt2.mean())/2

    mean_pixel_ref = pixel_ref1.mean()
    mean_pixel_tgt = pixel_tgt1.mean()
    bump_val = np.abs(mean_pixel_ref - mean_pixel_tgt)

    for slice in av1_batch3:

        try:

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

        except Exception as ex:
            print(ex)



if __name__ == '__main__':
    norm_batch4()
    #norm_batch1()
    #norm_batch2()
    #norm_batch3()
    #print_batch_graphs()