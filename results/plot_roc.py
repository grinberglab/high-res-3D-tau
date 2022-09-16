import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc



def main():

    # #AT100
    # test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
    # val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
    # fig1_name = '/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_ROC.png'
    #
    # auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    # auc_val = auc(val_stats[:, 5], val_stats[:, 3])
    #
    # plt.figure()
    # lw = 2
    # plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='Testing ROC (AUC {:.2f})'.format(auc_test))
    # plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Validation ROC (AUC {:.2f})'.format(auc_val))
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
    # #EoL with all AVID
    # test_stats = np.load('/home/maryana/storage2/Posdoc/End_of_Life/result/testing/End_of_Life_testing_stats.py.npy')
    # val_stats = np.load('/home/maryana/storage2/Posdoc/End_of_Life/result/validation/End_of_Life_validation_stats.py.npy')
    # fig2_name = '/home/maryana/storage2/Posdoc/End_of_Life/result/EoL_ROC.png'
    #
    # auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    # auc_val = auc(val_stats[:, 5], val_stats[:, 3])
    #
    # plt.figure()
    # lw = 2
    # plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='Testing ROC (AUC {:.2f})'.format(auc_test))
    # plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Validation ROC (AUC {:.2f})'.format(auc_val))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('EoL Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    # EoL with AT8
    # test_stats = np.load('/home/maryana/storage2/Posdoc/EoL+AT8/result/testing/heat_map/EoL_AT8_testing_stat.npy')
    #val_stats = np.load('/home/maryana/storage2/Posdoc/EoL+AT8/result/validation/heat_map/EoL_AT8_validation_stat.npy')
    #fig3_name = '/home/maryana/storage2/Posdoc/EoL+AT8/result/EoL_AT8_ROC.png'

    #auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    #auc_val = auc(val_stats[:, 5], val_stats[:, 3])

    #plt.figure()
    #lw = 2
    #plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange', lw=lw, label='Testing ROC (AUC {:.2f})'.format(auc_test))
    #plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Validation ROC (AUC {:.2f})'.format(auc_val))
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('EoL with AT8 Receiver operating characteristic')
    #plt.legend(loc="lower right")
    #plt.show()

    # EoL only
    #test_stats = np.load('/home/maryana/storage/Posdoc/EoL_only/result/testing/heat_map/EoL_only_testing_stat.npy')
    #val_stats = np.load('/home/maryana/storage/Posdoc/EoL_only/result/validation/heat_map/EoL_only_validation_stat.npy')
    #fig4_name = '/home/maryana/storage/Posdoc/EoL_only/result/EoL_only_ROC.png'

    #auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    #auc_val = auc(val_stats[:, 5], val_stats[:, 3])

    #plt.figure()
    #lw = 2
    #plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange', lw=lw, label='Testing ROC (AUC {:.2f})'.format(auc_test))
    #plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Validation ROC (AUC {:.2f})'.format(auc_val))
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('EoL only Receiver operating characteristic')
    #plt.legend(loc="lower right")
    #plt.show()

    # EoL_CP13+MAOB+AV_AT8_sgd_200_epoch
    test_stats = np.load('/home/maryana/storage2/Posdoc/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch/result/testing/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch/result/validation/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch_validation_stats.npy')
    fig5_name = '/home/maryana/storage2/Posdoc/EoL+AT8/result/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch.png'

    auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    auc_val = auc(val_stats[:, 5], val_stats[:, 3])

    plt.figure()
    lw = 2
    plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange', lw=lw,
             label='Testing ROC (AUC {:.2f})'.format(auc_test))
    plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw,
             label='Validation ROC (AUC {:.2f})'.format(auc_val))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('EoL_CP13+MAOB+AV_AT8_sgd_200_epoch Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # AT8_combined_on_EoL_CP13+MAOB+AV_AT8_sgd_200_epoch
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/EOL_testing/AT8_Testing_EOL_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/EOL_validation/AT8_Validation_EOL_stats.npy')
    fig6_name = '/home/maryana/storage2/Posdoc/EoL+AT8/result/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch.png'

    auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    auc_val = auc(val_stats[:, 5], val_stats[:, 3])

    plt.figure()
    lw = 2
    plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange', lw=lw, label='Testing ROC (AUC {:.2f})'.format(auc_test))
    plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Validation ROC (AUC {:.2f})'.format(auc_val))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AT8_combined_on_EoL_CP13+MAOB+AV_AT8_sgd_200_epoch_ROC')
    plt.legend(loc="lower right")
    plt.show()

    #plt.savefig(fig2_name)

    # #MC1
    # test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    # val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')
    # fig2_name = '/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_ROC.png'
    #
    # auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    # auc_val = auc(val_stats[:, 5], val_stats[:, 3])
    #
    # plt.figure()
    # lw = 2
    # plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange',lw=lw, label='Testing ROC (AUC {:.2f})'.format(auc_test))
    # plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw, label='Validation ROC (AUC {:.2f})'.format(auc_val))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('MC1 Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # plt.savefig(fig2_name)

    # # EoL with AT8 NEW
    # test_stats = np.load('/home/maryana/Desktop/EoL+AT8_new/result/AT8+EoL_testing_stat.npy')
    # val_stats = np.load('/home/maryana/Desktop/EoL+AT8_new/result/AT8+EoL_validation_stat.npy')
    # fig2_name = '/home/maryana/Desktop/EoL+AT8_new/result/AT8+EoL_new_roc.png'
    #
    # auc_test = auc(test_stats[:, 5], test_stats[:, 3])
    # auc_val = auc(val_stats[:, 5], val_stats[:, 3])
    #
    # plt.figure()
    # lw = 2
    # plt.plot(test_stats[:, 5], test_stats[:, 3], color='darkorange', lw=lw,
    #          label='Testing ROC (AUC {:.2f})'.format(auc_test))
    # plt.plot(val_stats[:, 5], val_stats[:, 3], color='purple', lw=lw,
    #          label='Validation ROC (AUC {:.2f})'.format(auc_val))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('EoL Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    pass


if __name__ == '__main__':
    main()