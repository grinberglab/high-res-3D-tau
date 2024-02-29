import sys
import numpy as np
import matplotlib.pyplot as plt
from convnet.util.scores import *
import glob
import os
import skimage.io as io
from skimage import img_as_ubyte

class Scores:

    # convert color-coded label to binary mask
    def _convert_mask2binary(self,g_truth):
        # process mask
        g_truth = img_as_ubyte(g_truth)
        mask_bkg = g_truth[..., 2] == 10
        mask_gm = g_truth[..., 2] == 255
        mask_thread = g_truth[..., 2] == 130
        mask_cell = g_truth[..., 2] == 0

        mask_fore = (mask_cell | mask_thread) * 255
        mask_bkg = (mask_gm | mask_bkg) * 255  # black background and GM together
        mask_fore_bin = mask_fore >= 255
        mask_back_bin = mask_bkg >= 255

        mask = np.concatenate((mask_fore_bin[..., np.newaxis], mask_back_bin[..., np.newaxis]), axis=2)
        # mask[mask > 1] = 0
        # mask[mask < 0] = 2
        return mask

    # get image/label pairs
    def _get_file_pairs(self,seg_dir, mask_dir):
        file_map = {}
        seg_list = glob.glob(os.path.join(seg_dir, '*.npy'))
        for prob_file in seg_list:
            basename = os.path.basename(prob_file)
            parcname = basename[:-9]
            mask_name = parcname + '_mask.tif'  # ground truth mask, not segmented mask
            mask_name = os.path.join(mask_dir, mask_name)

            file_map[prob_file] = mask_name

        return file_map

    # compute a set of scores based on the image/label pair
    def _compute_stats(self,file_map, prob_thres):
        dice = 0
        prec = 0
        rec = 0
        f1 = 0
        fpr = 0

        nFiles = len(file_map.keys())
        for prob_file in file_map.keys():

            prob_map = np.load(prob_file)
            seg_mask = prob_map >= prob_thres

            gt_file = file_map[prob_file]
            gd_truth = io.imread(gt_file)
            if gd_truth.ndim > 2:  # multichannel mask must be converted
                gd_truth = self._convert_mask2binary(gd_truth)
            
            x = gd_truth > 1
            y = gd_truth < 1
            gd_truth[x] = 0
            gd_truth[y] = 1
            # np.save("/home/grinberg/Downloads/CNN_test/results/testing/1.npy", gd_truth)

            dice += dice_coef_simple(gd_truth, seg_mask)  # gd_truth[...,0] == foreground

            # print(seg_mask.shape)
            # print(gd_truth.shape)
            
            prec += precision(gd_truth, seg_mask)
            rec += recall(gd_truth, seg_mask)
            f1 += F1(prec, rec)
            fpr += FPR(gd_truth, seg_mask)

        dice /= nFiles
        prec /= nFiles
        rec /= nFiles
        f1 /= nFiles
        fpr /= nFiles

        return dice, prec, rec, f1, fpr

    def run_compute_scores(self,seg_dir, gt_dir, out_file):
        file_map = self._get_file_pairs(seg_dir, gt_dir)

        probs = np.linspace(1, 0, num=20)

        n_probs = len(probs)
        dice_arr = np.zeros(n_probs)
        prec_arr = np.zeros(n_probs)
        rec_arr = np.zeros(n_probs)
        f1_arr = np.zeros(n_probs)
        fpr_arr = np.zeros(n_probs)

        for i in range(n_probs):
            prob = probs[i]

            print('Probability threshold: {:.2f}'.format(prob))

            dice, prec, rec, f1, fpr = self._compute_stats(file_map, prob)
            dice_arr[i] = dice
            prec_arr[i] = prec
            rec_arr[i] = rec
            f1_arr[i] = f1
            fpr_arr[i] = fpr

        print('Saving {}'.format(out_file))
        out_arr = np.concatenate((dice_arr[:,np.newaxis],prec_arr[:,np.newaxis],
                                rec_arr[:,np.newaxis],f1_arr[:,np.newaxis],fpr_arr[:,np.newaxis]),axis=1)
        np.save(out_file,out_arr)

        #plot ROC
        plt.figure()
        lw = 2
        plt.plot(fpr_arr, rec_arr, color='darkorange',
                lw=lw, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
        #plot precrec
        plt.figure()
        lw = 2
        plt.plot(rec_arr, prec_arr, color='darkorange',lw=lw, label='PR Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.show()



def main():
    if len(sys.argv) != 4:
        print('Usage: compute_convnet_stats.py <segmentation_dir> <ground_truth_dir> <output_stats_file>')
        exit()

    seg_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    out_file = sys.argv[3]

    compute_scores = Scores()
    scores = compute_scores.run_compute_scores(seg_dir,gt_dir,out_file)
    pass



if __name__ == '__main__':
    main()