
import numpy as np


def dice_coef2(gt,seg):
    k = 255
    dice = np.sum(seg[gt == k] == k) * 2.0 / (np.sum(seg[seg == k] == k) + np.sum(gt[gt == k] == k))

    return dice


def dice_coef_simple(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    idx_y_true = np.nonzero(y_true_f > 0)
    idx_y_pred = np.nonzero(y_pred_f > 0)

    if np.sum(y_true_f) == 0 and np.sum(y_pred_f) == 0: #both are puse background
        return 1.

    # num. TP
    TP = np.intersect1d(idx_y_pred,idx_y_true)
    nTP = float(np.sum(TP))

    #num. FP
    FP = np.setdiff1d(idx_y_pred,idx_y_true)
    nFP = float(np.sum(FP))

    #num FN
    FN = np.setdiff1d(idx_y_true,idx_y_pred)
    nFN = float(np.sum(FN))

    D = nTP/(nTP+nFP+nFN)
    D = 2.*D

    return D

def IoU(y_true,y_pred):
    nClasses = y_true.shape[2]
    iou = np.zeros((nClasses))
    for c in range(nClasses):
        pred = y_pred[...,c]
        gtruth = y_true[...,c]
        pred = pred.flatten()
        gtruth = gtruth.flatten()
        idx_y_true = np.nonzero(gtruth > 0)
        idx_y_pred = np.nonzero(pred > 0)

        if np.sum(pred) == 0 and np.sum(gtruth) == 0:  # both are pure background
            iou[c] = 1.
            continue

        # num. TP
        TP = np.intersect1d(idx_y_pred, idx_y_true)
        nTP = float(np.sum(TP))

        # num. FP
        FP = np.setdiff1d(idx_y_pred, idx_y_true)
        nFP = float(np.sum(FP))

        # num FN
        FN = np.setdiff1d(idx_y_true, idx_y_pred)
        nFN = float(np.sum(FN))

        D = nTP / (nTP + nFP + nFN)

        iou[c] = D

    return np.mean(iou),iou

def precision(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    idx_y_true = np.nonzero(y_true_f > 0)
    idx_y_pred = np.nonzero(y_pred_f > 0)

    if np.sum(y_true_f) == 0 and np.sum(y_pred_f) == 0: #both are puse background
        return 1.

    # num. TP
    TP = np.intersect1d(idx_y_pred,idx_y_true)
    nTP = float(len(TP))

    #num. FP
    FP = np.setdiff1d(idx_y_pred,idx_y_true)
    nFP = float(len(FP))

    try:
        P = nTP / (nTP + nFP)
    except ZeroDivisionError:
        P = 0

    return P


def recall(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    idx_y_true = np.nonzero(y_true_f > 0)
    idx_y_pred = np.nonzero(y_pred_f > 0)

    if np.sum(y_true_f) == 0 and np.sum(y_pred_f) == 0: #both are puse background
        return 1.

    # num. TP
    TP = np.intersect1d(idx_y_pred,idx_y_true)
    nTP = float(len(TP))

    #num FN
    FN = np.setdiff1d(idx_y_true,idx_y_pred)
    nFN = float(len(FN))

    try:
        R = nTP / (nTP + nFN)
    except ZeroDivisionError:
        R = 0

    return R

def F1(P,R):
    P = float(P)
    R = float(R)
    try:
        F1 = 2 * ((P * R) / (P + R))
    except ZeroDivisionError:
        F1 = 0

    return F1


def FPR(y_true,y_pred):
    y_true_n = 1 - y_true
    y_pred_n = 1 - y_pred
    y_true_n_f = y_true_n.flatten()
    y_pred_n_f = y_pred_n.flatten()
    idx_y_n_true = np.nonzero(y_true_n_f > 0)
    idx_y_n_pred = np.nonzero(y_pred_n_f > 0)

    TN = np.intersect1d(idx_y_n_pred,idx_y_n_true) #index of true negative elements
    nTN = float(len(TN))

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    idx_y_true = np.nonzero(y_true_f > 0)
    idx_y_pred = np.nonzero(y_pred_f > 0)

    #num. FP
    FP = np.setdiff1d(idx_y_pred,idx_y_true)
    nFP = float(len(FP))

    FPR = nFP/(nFP+nTN)

    return FPR





def TN_rate(y_true,y_pred):
    y_true = 1 - y_true
    y_pred = 1 - y_pred
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    idx_y_true = np.nonzero(y_true_f > 0)
    idx_y_pred = np.nonzero(y_pred_f > 0)

    TN = np.intersect1d(idx_y_pred,idx_y_true)
    nTN = float(len(TN))

    return nTN
