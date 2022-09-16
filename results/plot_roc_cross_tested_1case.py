import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc



def main():

    #AT100 Case#1 segmented using model trained with AT100 Case#2
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
    case1_2_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results_trained_with_AV2/testing/stats_AV1_AV2_testing_stats.npy')
    case1_2_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results_trained_with_AV2/validation/stats_AV1_AV2_validation_stats.npy')
    case2_1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT100/results_trained_with_AV1/testing/stats_AV2_AV1_testing_stats.npy')
    case2_1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT100/results_trained_with_AV1/validation/stats_AV2_AV1_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results_trained_with_AV2/AT100_cross_tested_ROC.png'

    auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    auc_val = auc(val_stats[:, 5], val_stats[:, 3])

    case1_2_auc_test = auc(case1_2_test_stats[:, 5], case1_2_test_stats[:, 3])
    case1_2_auc_val = auc(case1_2_val_stats[:, 5], case1_2_val_stats[:, 3])
    case2_1_auc_test = auc(case2_1_test_stats[:, 5], case2_1_test_stats[:, 3])
    case2_1_auc_val = auc(case2_1_val_stats[:, 5], case2_1_val_stats[:, 3])

    plt.figure()
    lw = 2
    plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='Full testing (AUC {:.4f})'.format(auc_test))
    plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Full validation (AUC {:.4f})'.format(auc_val))

    plt.plot(case1_2_test_stats[:, 5], case1_2_test_stats[:, 3],'--', color='red',lw=lw, label='Case#1/Case#2 testing (AUC {:.4f})'.format(case1_2_auc_test))
    plt.plot(case1_2_val_stats[:, 5], case1_2_val_stats[:, 3],'--', color='blue', lw=lw, label='Case#1/Case#2 validation (AUC {:.4f})'.format(case1_2_auc_val))
    plt.plot(case2_1_test_stats[:, 5], case2_1_test_stats[:, 3],'--', color='green',lw=lw, label='Case#2/Case#1 testing (AUC {:.4f})'.format(case2_1_auc_test))
    plt.plot(case2_1_val_stats[:, 5], case2_1_val_stats[:, 3],'--', color='peru', lw=lw, label='Case#2/Case#1 validation (AUC {:.4f})'.format(case2_1_auc_val))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AT100 cross-testing receiver operating characteristic')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)


    #AT8 segmented using model trained with AT100
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/testing/AT8_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_validation_stats.npy')
    case1_2_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results_trained_with_AV2/testing/stats_AV1_AV2_testing_stats.npy')
    case1_2_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results_trained_with_AV2/validation/stats_AV1_AV2_validation_stats.npy')
    case2_1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT8/results_trained_with_AV1/testing/stats_AV2_AV1_testing_stats.npy')
    case2_1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT8/results_trained_with_AV1/validation/stats_AV2_AV1_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results_trained_with_AV2/AT8_cross_tested_ROC.png'

    auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    auc_val = auc(val_stats[:, 5], val_stats[:, 3])

    case1_2_auc_test = auc(case1_2_test_stats[:, 5], case1_2_test_stats[:, 3])
    case1_2_auc_val = auc(case1_2_val_stats[:, 5], case1_2_val_stats[:, 3])
    case2_1_auc_test = auc(case2_1_test_stats[:, 5], case2_1_test_stats[:, 3])
    case2_1_auc_val = auc(case2_1_val_stats[:, 5], case2_1_val_stats[:, 3])

    plt.figure()
    lw = 2
    plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='Full testing (AUC {:.4f})'.format(auc_test))
    plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Full validation (AUC {:.4f})'.format(auc_val))

    plt.plot(case1_2_test_stats[:, 5], case1_2_test_stats[:, 3],'--', color='red',lw=lw, label='Case#1/Case#2 testing (AUC {:.4f})'.format(case1_2_auc_test))
    plt.plot(case1_2_val_stats[:, 5], case1_2_val_stats[:, 3],'--', color='blue', lw=lw, label='Case#1/Case#2 validation (AUC {:.4f})'.format(case1_2_auc_val))
    plt.plot(case2_1_test_stats[:, 5], case2_1_test_stats[:, 3],'--', color='green',lw=lw, label='Case#2/Case#1 testing (AUC {:.4f})'.format(case2_1_auc_test))
    plt.plot(case2_1_val_stats[:, 5], case2_1_val_stats[:, 3],'--', color='peru', lw=lw, label='Case#2/Case#1 validation (AUC {:.4f})'.format(case2_1_auc_val))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AT8 cross-testing receiver operating characteristic')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)

    #MC1 segmented using model trained with AT100
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')
    case1_2_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results_trained_with_AV2/testing/stats_AV1_AV2_testing_stats.npy')
    case1_2_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results_trained_with_AV2/validation/stats_AV1_AV2_validation_stats.npy')
    case2_1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/MC1/results_trained_with_AV1/testing/stats_AV2_AV1_testing_stats.npy')
    case2_1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/MC1/results_trained_with_AV1/validation/stats_AV2_AV1_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results_trained_with_AV2/MC1_cross_tested_ROC.png'

    auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    auc_val = auc(val_stats[:, 5], val_stats[:, 3])

    case1_2_auc_test = auc(case1_2_test_stats[:, 5], case1_2_test_stats[:, 3])
    case1_2_auc_val = auc(case1_2_val_stats[:, 5], case1_2_val_stats[:, 3])
    case2_1_auc_test = auc(case2_1_test_stats[:, 5], case2_1_test_stats[:, 3])
    case2_1_auc_val = auc(case2_1_val_stats[:, 5], case2_1_val_stats[:, 3])

    plt.figure()
    lw = 2
    plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='Full testing (AUC {:.4f})'.format(auc_test))
    plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Full validation (AUC {:.4f})'.format(auc_val))

    plt.plot(case1_2_test_stats[:, 5], case1_2_test_stats[:, 3],'--', color='red',lw=lw, label='Case#1/Case#2 testing (AUC {:.4f})'.format(case1_2_auc_test))
    plt.plot(case1_2_val_stats[:, 5], case1_2_val_stats[:, 3],'--', color='blue', lw=lw, label='Case#1/Case#2 validation (AUC {:.4f})'.format(case1_2_auc_val))
    plt.plot(case2_1_test_stats[:, 5], case2_1_test_stats[:, 3],'--', color='green',lw=lw, label='Case#2/Case#1 testing (AUC {:.4f})'.format(case2_1_auc_test))
    plt.plot(case2_1_val_stats[:, 5], case2_1_val_stats[:, 3],'--', color='peru', lw=lw, label='Case#2/Case#1 validation (AUC {:.4f})'.format(case2_1_auc_val))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MC1 cross-testing receiver operating characteristic')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)

    pass


if __name__ == '__main__':
    main()