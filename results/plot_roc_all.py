import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

#roc curves for all "full models" together
#"full model" stands for a model trained with all images patches together. I.e All AT100 (av1 + av2)

def main():

    #All markers together
    AT100_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
    AT100_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
    AT8_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/testing/AT8_testing_stats.npy')
    AT8_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_validation_stats.npy')
    MC1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    MC1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/All_ROC.png'

    AT100_auc_test = auc(AT100_test_stats[:, 5], AT100_test_stats[:, 3])
    AT100_auc_val = auc(AT100_val_stats[:, 5], AT100_val_stats[:, 3])
    AT8_auc_test = auc(AT8_test_stats[:, 5], AT8_test_stats[:, 3])
    AT8_auc_val = auc(AT8_val_stats[:, 5], AT8_val_stats[:, 3])
    MC1_auc_test = auc(MC1_test_stats[:, 5], MC1_test_stats[:, 3])
    MC1_auc_val = auc(MC1_val_stats[:, 5], MC1_val_stats[:, 3])

    plt.figure()
    lw = 2
    plt.plot(AT100_test_stats[:, 5], AT100_test_stats[:, 3],'--', color='red',lw=lw, label='AT100 testing (AUC {:.2f})'.format(AT100_auc_test))
    plt.plot(AT100_val_stats[:, 5], AT100_val_stats[:, 3], color='red', lw=lw, label='AT100 validation (AUC {:.2f})'.format(AT100_auc_val))
    plt.plot(AT8_test_stats[:, 5], AT8_test_stats[:, 3],'--', color='green',lw=lw, label='AT8 testing ROC (AUC {:.2f})'.format(AT8_auc_test))
    plt.plot(AT8_val_stats[:, 5], AT8_val_stats[:, 3], color='green', lw=lw, label='AT8 validation ROC (AUC {:.2f})'.format(AT8_auc_val))
    plt.plot(MC1_test_stats[:, 5], MC1_test_stats[:, 3],'--', color='blue',lw=lw, label='MC1 testing ROC (AUC {:.2f})'.format(MC1_auc_test))
    plt.plot(MC1_val_stats[:, 5], MC1_val_stats[:, 3], color='blue', lw=lw, label='MC1 validation ROC (AUC {:.2f})'.format(MC1_auc_val))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(fig1_name)

    pass


if __name__ == '__main__':
    main()