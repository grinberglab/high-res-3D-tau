import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc



def main():

    # #AT100
    #
    # fig1_name = '/home/maryana/storage2/Posdoc/AVID/AT100/results/AT100_ROC_all.png'
    #
    # test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
    # val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
    # AV13_AT100_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results/testing/AV13_AT100_testing_stats.npy')
    # AV13_AT100_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results/validation/AV13_AT100_validation_stats.npy')
    # AV23_AT100_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT100/results/testing/AV23_AT100_testing_stats.npy')
    # AV23_AT100_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT100/results/validation/AV23_AT100_validation_stats.npy')
    #
    # auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    # auc_val = auc(val_stats[:, 5], val_stats[:, 3])
    #
    # AV13_AT100_auc_test = auc(AV13_AT100_test_stats[:, 5], AV13_AT100_test_stats[:, 3])
    # AV13_AT100_auc_val = auc(AV13_AT100_val_stats[:, 5], AV13_AT100_val_stats[:, 3])
    #
    # AV23_AT100_auc_test = auc(AV23_AT100_test_stats[:, 5], AV23_AT100_test_stats[:, 3])
    # AV23_AT100_auc_val = auc(AV23_AT100_val_stats[:, 5], AV23_AT100_val_stats[:, 3])
    #
    # plt.figure()
    # lw = 2
    #
    # plt.plot(AV13_AT100_test_stats[:, 5], AV13_AT100_test_stats[:, 3],'--', color='red',lw=lw, label='#1 Test (AUC {:.4f})'.format(AV13_AT100_auc_test))
    # plt.plot(AV13_AT100_val_stats[:, 5], AV13_AT100_val_stats[:, 3],'--', color='green', lw=lw, label='#1 Validation (AUC {:.4f})'.format(AV13_AT100_auc_val))
    # plt.plot(AV23_AT100_test_stats[:, 5], AV23_AT100_test_stats[:, 3],'--', color='blue',lw=lw, label='#2 Test (AUC {:.4f})'.format(AV23_AT100_auc_test))
    # plt.plot(AV23_AT100_val_stats[:, 5], AV23_AT100_val_stats[:, 3],'--', color='peru', lw=lw, label='#2 Validation (AUC {:.4f})'.format(AV23_AT100_auc_val))
    #
    # plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='All Test (AUC {:.4f})'.format(auc_test))
    # plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='All Validation (AUC {:.4f})'.format(auc_val))
    #
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('AT100 Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # plt.savefig(fig1_name)
    #
    #
    # #AT8
    #
    # fig2_name = '/home/maryana/storage2/Posdoc/AVID/AT8/results/AT8_ROC_all.png'
    #
    # test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/testing/AT8_testing_stats.npy')
    # val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_validation_stats.npy')
    #
    # AV13_AT8_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results/testing/AV13_AT8_testing_stats.npy')
    # AV13_AT8_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results/validation/AV13_AT8_validation_stats.npy')
    # AV23_AT8_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT8/results/testing/AV23_AT8_testing_stats.npy')
    # AV23_AT8_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT8/results/validation/AV23_AT8_validation_stats.npy')
    #
    # auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    # auc_val = auc(val_stats[:, 5], val_stats[:, 3])
    #
    # AV13_AT8_auc_test = auc(AV13_AT8_test_stats[:, 5], AV13_AT8_test_stats[:, 3])
    # AV13_AT8_auc_val = auc(AV13_AT8_val_stats[:, 5], AV13_AT8_val_stats[:, 3])
    # AV23_AT8_auc_test = auc(AV23_AT8_test_stats[:, 5], AV23_AT8_test_stats[:, 3])
    # AV23_AT8_auc_val = auc(AV23_AT8_val_stats[:, 5], AV23_AT8_val_stats[:, 3])
    #
    # plt.figure()
    # lw = 2
    #
    # plt.plot(AV13_AT8_test_stats[:, 5], AV13_AT8_test_stats[:, 3],'--', color='red',lw=lw, label='#1 Test (AUC {:.4f})'.format(AV13_AT8_auc_test))
    # plt.plot(AV13_AT8_val_stats[:, 5], AV13_AT8_val_stats[:, 3],'--', color='green', lw=lw, label='#1 Validation (AUC {:.4f})'.format(AV13_AT8_auc_val))
    # plt.plot(AV23_AT8_test_stats[:, 5], AV23_AT8_test_stats[:, 3],'--', color='blue',lw=lw, label='#2 Test (AUC {:.4f})'.format(AV23_AT8_auc_test))
    # plt.plot(AV23_AT8_val_stats[:, 5], AV23_AT8_val_stats[:, 3],'--', color='peru', lw=lw, label='#2 Validation (AUC {:.4f})'.format(AV23_AT8_auc_val))
    #
    # plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='All Test (AUC {:.4f})'.format(auc_test))
    # plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='All Validation (AUC {:.4f})'.format(auc_val))
    #
    #
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('AT8 Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # plt.savefig(fig2_name)


    # MC1

    fig1_name = '/home/maryana/storage2/Posdoc/AVID/MC1/results/MC1_roc_all.png'

    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')
    AV13_MC1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results/testing/AV13_MC1_testing_stats.npy')
    AV13_MC1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results/validation/AV13_MC1_validation_stats.npy')
    AV23_MC1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/MC1/results/testing/AV23_MC1_testing_stats.npy')
    AV23_MC1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/MC1/results/validation/AV23_MC1_validation_stats.npy')


    auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    auc_val = auc(val_stats[:, 5], val_stats[:, 3])

    AV13_MC1_auc_test = auc(AV13_MC1_test_stats[:, 5], AV13_MC1_test_stats[:, 3])
    AV13_MC1_auc_val = auc(AV13_MC1_val_stats[:, 5], AV13_MC1_val_stats[:, 3])
    AV23_MC1_auc_test = auc(AV23_MC1_test_stats[:, 5], AV23_MC1_test_stats[:, 3])
    AV23_MC1_auc_val = auc(AV23_MC1_val_stats[:, 5], AV23_MC1_val_stats[:, 3])

    plt.figure()
    lw = 2

    plt.plot(AV13_MC1_test_stats[:, 5], AV13_MC1_test_stats[:, 3],'--', color='red',lw=lw, label='#1 Test (AUC {:.4f})'.format(AV13_MC1_auc_test))
    plt.plot(AV13_MC1_val_stats[:, 5], AV13_MC1_val_stats[:, 3],'--', color='green', lw=lw, label='#1 Validation (AUC {:.4f})'.format(AV13_MC1_auc_val))
    plt.plot(AV13_MC1_test_stats[:, 5], AV13_MC1_test_stats[:, 3],'--', color='blue',lw=lw, label='#2 Test (AUC {:.4f})'.format(AV23_MC1_auc_test))
    plt.plot(AV13_MC1_val_stats[:, 5], AV13_MC1_val_stats[:, 3],'--', color='peru', lw=lw, label='#2 Validation (AUC {:.4f})'.format(AV23_MC1_auc_val))

    plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='All Test (AUC {:.4f})'.format(auc_test))
    plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='All Validation (AUC {:.4f})'.format(auc_val))


    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MC1 Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(fig1_name)





    pass


if __name__ == '__main__':
    main()