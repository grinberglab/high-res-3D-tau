import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc


def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx


def main():

    thres = 0.5

    #AT100 Case#1 segmented using model trained with AT100 Case#2 and vice-versa
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
    case1_2_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results_trained_with_AV2/testing/stats_AV1_AV2_testing_stats.npy')
    case1_2_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results_trained_with_AV2/validation/stats_AV1_AV2_validation_stats.npy')
    case2_1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT100/results_trained_with_AV1/testing/stats_AV2_AV1_testing_stats.npy')
    case2_1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT100/results_trained_with_AV1/validation/stats_AV2_AV1_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results_trained_with_AV2/AT100_cross_tested_precrec.png'

    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]
    # case1_2_prec_t = case1_2_test_stats[:, 2]; case1_2_recall_t = case1_2_test_stats[:, 3]
    # case1_2_prec_v = case1_2_val_stats[:, 2]; case1_2_recall_v = case1_2_val_stats[:, 3]
    # case2_1_prec_t = case2_1_test_stats[:, 2]; case2_1_recall_t = case2_1_test_stats[:, 3]
    # case2_1_prec_v = case2_1_val_stats[:, 2]; case2_1_recall_v = case2_1_val_stats[:, 3]

    auc_test = auc(test_stats[:, 3], test_stats[:, 2])
    auc_val = auc(val_stats[:, 3], val_stats[:, 2])
    case1_2_auc_test = auc(case1_2_test_stats[:, 3], case1_2_test_stats[:, 2])
    case1_2_auc_val = auc(case1_2_val_stats[:, 3], case1_2_val_stats[:, 2])
    case2_1_auc_test = auc(case2_1_test_stats[:, 3], case2_1_test_stats[:, 2])
    case2_1_auc_val = auc(case2_1_val_stats[:, 3], case2_1_val_stats[:, 2])

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]


    plt.figure()
    lw = 2
    plt.plot(test_stats[:, 3], test_stats[:, 2], color='darkorange',lw=lw, label='Full testing (AUC {:.4f})'.format(auc_test))
    plt.plot(val_stats[:, 3], val_stats[:, 2], color='purple', lw=lw, label='Full validation (AUC {:.4f})'.format(auc_val))

    plt.plot(case1_2_test_stats[:, 3], case1_2_test_stats[:, 2],'--', color='red',lw=lw, label='Case#1/Case#2 testing (AUC {:.4f})'.format(case1_2_auc_test))
    plt.plot(case1_2_val_stats[:, 3], case1_2_val_stats[:, 2],'--', color='blue', lw=lw, label='Case#1/Case#2 validation (AUC {:.4f})'.format(case1_2_auc_val))
    plt.plot(case2_1_test_stats[:, 3], case2_1_test_stats[:, 2],'--', color='green',lw=lw, label='Case#2/Case#1 testing (AUC {:.4f})'.format(case2_1_auc_test))
    plt.plot(case2_1_val_stats[:, 3], case2_1_val_stats[:, 2],'--', color='peru', lw=lw, label='Case#2/Case#1 validation (AUC {:.4f})'.format(case2_1_auc_val))

    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AT100 cross-testing precision-recall')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)



    #AT8 Case#1 segmented using model trained with AT100 Case#2 and vice-versa
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/testing/AT8_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_validation_stats.npy')
    case1_2_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results_trained_with_AV2/testing/stats_AV1_AV2_testing_stats.npy')
    case1_2_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results_trained_with_AV2/validation/stats_AV1_AV2_validation_stats.npy')
    case2_1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT8/results_trained_with_AV1/testing/stats_AV2_AV1_testing_stats.npy')
    case2_1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT8/results_trained_with_AV1/validation/stats_AV2_AV1_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results_trained_with_AV2/AT8_cross_tested_precrec.png'

    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]
    # case1_2_prec_t = case1_2_test_stats[:, 2]; case1_2_recall_t = case1_2_test_stats[:, 3]
    # case1_2_prec_v = case1_2_val_stats[:, 2]; case1_2_recall_v = case1_2_val_stats[:, 3]
    # case2_1_prec_t = case2_1_test_stats[:, 2]; case2_1_recall_t = case2_1_test_stats[:, 3]
    # case2_1_prec_v = case2_1_val_stats[:, 2]; case2_1_recall_v = case2_1_val_stats[:, 3]

    auc_test = auc(test_stats[:, 3], test_stats[:, 2])
    auc_val = auc(val_stats[:, 3], val_stats[:, 2])
    case1_2_auc_test = auc(case1_2_test_stats[:, 3], case1_2_test_stats[:, 2])
    case1_2_auc_val = auc(case1_2_val_stats[:, 3], case1_2_val_stats[:, 2])
    case2_1_auc_test = auc(case2_1_test_stats[:, 3], case2_1_test_stats[:, 2])
    case2_1_auc_val = auc(case2_1_val_stats[:, 3], case2_1_val_stats[:, 2])

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]


    plt.figure()
    lw = 2
    plt.plot(test_stats[:, 3], test_stats[:, 2], color='darkorange',lw=lw, label='Full testing (AUC {:.4f})'.format(auc_test))
    plt.plot(val_stats[:, 3], val_stats[:, 2], color='purple', lw=lw, label='Full validation (AUC {:.4f})'.format(auc_val))

    plt.plot(case1_2_test_stats[:, 3], case1_2_test_stats[:, 2],'--', color='red',lw=lw, label='Case#1/Case#2 testing (AUC {:.4f})'.format(case1_2_auc_test))
    plt.plot(case1_2_val_stats[:, 3], case1_2_val_stats[:, 2],'--', color='blue', lw=lw, label='Case#1/Case#2 validation (AUC {:.4f})'.format(case1_2_auc_val))
    plt.plot(case2_1_test_stats[:, 3], case2_1_test_stats[:, 2],'--', color='green',lw=lw, label='Case#2/Case#1 testing (AUC {:.4f})'.format(case2_1_auc_test))
    plt.plot(case2_1_val_stats[:, 3], case2_1_val_stats[:, 2],'--', color='peru', lw=lw, label='Case#2/Case#1 validation (AUC {:.4f})'.format(case2_1_auc_val))

    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AT8 cross-testing precision-recall')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)


    #MC1 Case#1 segmented using model trained with AT100 Case#2 and vice-versa
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')
    case1_2_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results_trained_with_AV2/testing/stats_AV1_AV2_testing_stats.npy')
    case1_2_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results_trained_with_AV2/validation/stats_AV1_AV2_validation_stats.npy')
    case2_1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/MC1/results_trained_with_AV1/testing/stats_AV2_AV1_testing_stats.npy')
    case2_1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/MC1/results_trained_with_AV1/validation/stats_AV2_AV1_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results_trained_with_AV2/MC1_cross_tested_precrec.png'

    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]
    # case1_2_prec_t = case1_2_test_stats[:, 2]; case1_2_recall_t = case1_2_test_stats[:, 3]
    # case1_2_prec_v = case1_2_val_stats[:, 2]; case1_2_recall_v = case1_2_val_stats[:, 3]
    # case2_1_prec_t = case2_1_test_stats[:, 2]; case2_1_recall_t = case2_1_test_stats[:, 3]
    # case2_1_prec_v = case2_1_val_stats[:, 2]; case2_1_recall_v = case2_1_val_stats[:, 3]

    auc_test = auc(test_stats[:, 3], test_stats[:, 2])
    auc_val = auc(val_stats[:, 3], val_stats[:, 2])
    case1_2_auc_test = auc(case1_2_test_stats[:, 3], case1_2_test_stats[:, 2])
    case1_2_auc_val = auc(case1_2_val_stats[:, 3], case1_2_val_stats[:, 2])
    case2_1_auc_test = auc(case2_1_test_stats[:, 3], case2_1_test_stats[:, 2])
    case2_1_auc_val = auc(case2_1_val_stats[:, 3], case2_1_val_stats[:, 2])

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]


    plt.figure()
    lw = 2
    plt.plot(test_stats[:, 3], test_stats[:, 2], color='darkorange',lw=lw, label='Full testing (AUC {:.4f})'.format(auc_test))
    plt.plot(val_stats[:, 3], val_stats[:, 2], color='purple', lw=lw, label='Full validation (AUC {:.4f})'.format(auc_val))

    plt.plot(case1_2_test_stats[:, 3], case1_2_test_stats[:, 2],'--', color='red',lw=lw, label='Case#1/Case#2 testing (AUC {:.4f})'.format(case1_2_auc_test))
    plt.plot(case1_2_val_stats[:, 3], case1_2_val_stats[:, 2],'--', color='blue', lw=lw, label='Case#1/Case#2 validation (AUC {:.4f})'.format(case1_2_auc_val))
    plt.plot(case2_1_test_stats[:, 3], case2_1_test_stats[:, 2],'--', color='green',lw=lw, label='Case#2/Case#1 testing (AUC {:.4f})'.format(case2_1_auc_test))
    plt.plot(case2_1_val_stats[:, 3], case2_1_val_stats[:, 2],'--', color='peru', lw=lw, label='Case#2/Case#1 validation (AUC {:.4f})'.format(case2_1_auc_val))

    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('MC1 cross-testing precision-recall')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)



    pass


if __name__ == '__main__':
    main()