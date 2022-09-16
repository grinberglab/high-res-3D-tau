import glob
import os
import re
import nibabel as nib
import numpy as np
import pickle
from convnet.util.scores import dice_coef2


def run_dice(histo_dir,block_dir,results_file):
    histo_files = glob.glob(os.path.join(histo_dir,'*REG.nii'))
    nFiles = len(histo_files)

    dices = {}

    for histo_file in histo_files:

        print('Processing {}'.format(histo_file))

        basename = os.path.basename(histo_file)
        idx = [m.start() for m in re.finditer('_', basename)]
        id1 = idx[0]
        id2 = idx[1]
        id_str = basename[id1+1:id2]

        block_name = 'AV13_001-Whole_Brain_0{}_mask_right_hippocampus_dice_mask.nii'.format(id_str)
        block_file = os.path.join(block_dir,block_name)

        nii_histo = nib.load(histo_file)
        nii_block = nib.load(block_file)

        histo = nii_histo.get_data()
        block = nii_block.get_data()

        histo[histo > 0] = 255
        block[block > 0] = 255

        D = dice_coef2(block,histo)

        dices[basename] = D

    pickle_out = open(results_file,'wb')
    pickle.dump(dices,pickle_out)
    pickle_out.close()

    uDice = 0
    for k in dices.keys():
        uDice += dices[k]
    uDice /= nFiles
    print('Mean Dice: {}'.format(uDice))




if __name__ == '__main__':
    histo_dir = '/home/maryana/R_DRIVE/Experiments/1_AVID/Cases/1811-001/Master_Package_1181-001/Images/1181-001_BLOCKFACE/dice_masks/maks_to_right_hipocampus/done/AT100/masks'
    block_dir = '/home/maryana/R_DRIVE/Experiments/1_AVID/Cases/1811-001/Master_Package_1181-001/Images/1181-001_BLOCKFACE/dice_masks/maks_to_right_hipocampus/done/block/masks'
    out_file = '/home/maryana/R_DRIVE/Experiments/1_AVID/Cases/1811-001/Master_Package_1181-001/Images/1181-001_BLOCKFACE/dice_masks/maks_to_right_hipocampus/done/AT100/AT100_Dice_dict.pickle'
    run_dice(histo_dir,block_dir,out_file)