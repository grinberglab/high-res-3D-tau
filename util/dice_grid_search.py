import os
import glob
import numpy as np
from convnet.util.scores import dice_coef_simple,IoU,precision,recall,F1,FPR
import matplotlib.pyplot as plt


pred_dir = '/home/maryana/storage2/Posdoc/AVID/AV23/AT100/slidenet_2classes/training/images204'
mask_dir = '/home/maryana/storage2/Posdoc/AVID/AV23/AT100/slidenet_2classes/training/masks204/patches'
pred_list = glob.glob(os.path.join(pred_dir,'*.npy'))

nPatches = len(pred_list)

grid = np.arange(0.0,1.01,0.1)
nGrid = len(grid)
dice_arr = np.zeros((nGrid))
P_arr = np.zeros((nGrid))
R_arr = np.zeros((nGrid))
F1_arr = np.zeros((nGrid))
FPR_arr = np.zeros((nGrid))
best_dice = 0
best_th = 0
count = 0

for th in grid:

    mean_dice = 0
    mean_P = 0
    mean_R = 0
    mean_F1 = 0
    mean_FPR = 0
    for f in pred_list:
        pred = np.load(f)
        basename = os.path.basename(f)
        id = basename[5:-13]
        maskname = 'patch_mask' + id + '.npy'
        mask_path = os.path.join(mask_dir,maskname)
        mask = np.load(mask_path)
        #mask = mask[...,0]

        mask_fore = pred >= th
        mask_fore = (mask_fore * 1).astype('uint8')
        mask_bkg = 1-mask_fore

        mask_pred = np.concatenate((mask_fore[...,np.newaxis],mask_bkg[...,np.newaxis]),axis=2)

        #iou,iouc = IoU(mask,mask_pred)
        #mean_dice += iou
        P = precision(mask[...,0],mask_fore)
        R = recall(mask[...,0],mask_fore)
        f1 = F1(P,R)
        fpr = FPR(mask[...,0],mask_fore)

        mean_P += P
        mean_R += R
        mean_F1 += f1
        mean_FPR += fpr


    #mean_dice /= nPatches
    mean_P /= nPatches
    mean_R /= nPatches
    mean_F1 /= nPatches
    mean_FPR /= nPatches

    #dice_arr[count] = mean_F1
    P_arr[count] = mean_P
    R_arr[count] = mean_R
    F1_arr[count] = mean_F1
    FPR_arr[count] = mean_FPR
    # if mean_F1 >= best_dice:
    #     best_dice = mean_dice
    #     best_th = th

    count += 1


#print('Best threshold: {}, Best Dice {}'.format(best_th,best_dice))
plt.plot(P_arr,R_arr,F1_arr)
pass






