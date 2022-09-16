import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx


def main():

    thres = 0.5


    #AT100
    AT100_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
    AT100_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
    AT8_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/testing/AT8_testing_stats.npy')
    AT8_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_validation_stats.npy')
    MC1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    MC1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')

    fig1_name = '/home/maryana/storage2/Posdoc/AVID/All_precision_recall.png'

    AT100_prec_t = AT100_test_stats[:, 2]; AT100_recall_t = AT100_test_stats[:, 3]
    AT100_prec_v = AT100_val_stats[:, 2]; AT100_recall_v = AT100_val_stats[:, 3]
    AT8_prec_t = AT8_test_stats[:, 2]; AT8_recall_t = AT8_test_stats[:, 3]
    AT8_prec_v = AT8_val_stats[:, 2]; AT8_recall_v = AT8_val_stats[:, 3]
    MC1_prec_t = MC1_test_stats[:, 2]; MC1_recall_t = MC1_test_stats[:, 3]
    MC1_prec_v = MC1_val_stats[:, 2]; MC1_recall_v = MC1_val_stats[:, 3]

    AT100_fpr_t = AT100_test_stats[:,5]; AT100_f1_t = AT100_test_stats[:,4]
    AT100_fpr_v = AT100_val_stats[:,5]; AT100_f1_v = AT100_val_stats[:,4]
    AT8_fpr_t = AT8_test_stats[:,5]; AT8_f1_t = AT8_test_stats[:,4]
    AT8_fpr_v = AT8_val_stats[:,5]; AT8_f1_v = AT8_val_stats[:,4]
    MC1_fpr_t = MC1_test_stats[:,5]; MC1_f1_t = MC1_test_stats[:,4]
    MC1_fpr_v = MC1_val_stats[:,5]; MC1_f1_v = MC1_val_stats[:,4]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    print('AT100')
    print('Testing: {}(Prec) {}(TPR) {}(FPR) {}(F1) {}(TNR) {}(FNR)'.format(AT100_prec_t[index],AT100_recall_t[index],AT100_fpr_t[index],AT100_f1_t[index],1-AT100_fpr_t[index],1-AT100_recall_t[index]))
    print('Val: {}(Prec) {}(TPR) {}(FPR) {}(F1) {}(TNR) {}(FNR)'.format(AT100_prec_v[index],AT100_recall_v[index],AT100_fpr_v[index],AT100_f1_v[index],1-AT100_fpr_v[index],1-AT100_recall_v[index]))

    print('AT8')
    print('Testing: {}(Prec) {}(TPR) {}(FPR) {}(F1) {}(TNR) {}(FNR)'.format(AT8_prec_t[index],AT8_recall_t[index],AT8_fpr_t[index],AT8_f1_t[index],1-AT8_fpr_t[index],1-AT8_recall_t[index]))
    print('Val: {}(Prec) {}(TPR) {}(FPR) {}(F1) {}(TNR) {}(FNR)'.format(AT8_prec_v[index],AT8_recall_v[index],AT8_fpr_v[index],AT8_f1_v[index],1-AT8_fpr_v[index],1-AT8_recall_v[index]))

    print('MC1')
    print('Testing: {}(Prec) {}(TPR) {}(FPR) {}(F1) {}(TNR) {}(FNR)'.format(MC1_prec_t[index],MC1_recall_t[index],MC1_fpr_t[index],MC1_f1_t[index],1-MC1_fpr_t[index],1-MC1_recall_t[index]))
    print('Val: {}(Prec) {}(TPR) {}(FPR) {}(F1) {}(TNR) {}(FNR)'.format(MC1_prec_v[index],MC1_recall_v[index],MC1_fpr_v[index],MC1_f1_v[index],1-MC1_fpr_v[index],1-MC1_recall_v[index]))

    AT100_x_thres_t = AT100_recall_t[index]; AT100_y_thres_t = AT100_prec_t[index]
    AT100_x_thres_v = AT100_recall_v[index]; AT100_y_thres_v = AT100_prec_v[index]
    AT8_x_thres_t = AT8_recall_t[index]; AT8_y_thres_t = AT8_prec_t[index]
    AT8_x_thres_v = AT8_recall_v[index]; AT8_y_thres_v = AT8_prec_v[index]
    MC1_x_thres_t = MC1_recall_t[index]; MC1_y_thres_t = MC1_prec_t[index]
    MC1_x_thres_v = MC1_recall_v[index]; MC1_y_thres_v = MC1_prec_v[index]


    AT100_auc_t = auc(AT100_recall_t, AT100_prec_t); AT100_auc_v = auc(AT100_recall_v, AT100_prec_v)
    AT8_auc_t = auc(AT8_recall_t, AT8_prec_t); AT8_auc_v = auc(AT8_recall_v, AT8_prec_v)
    MC1_auc_t = auc(MC1_recall_t, MC1_prec_t); MC1_auc_v = auc(MC1_recall_v, MC1_prec_v)

    plt.figure()
    lw = 2
    plt.plot(AT100_recall_t,AT100_prec_t,'--', color='red',lw=lw, label='AT100 testing (AUC {:.2f})'.format(AT100_auc_t))
    plt.plot(AT100_recall_v,AT100_prec_v,  color='red', lw=lw, label='AT100 validation (AUC {:.2f})'.format(AT100_auc_v))
    plt.plot(AT100_x_thres_t, AT100_y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(AT100_x_thres_v, AT100_y_thres_v, color='red', lw=lw, marker='*', markersize=12)

    plt.plot(AT8_recall_t,AT8_prec_t,'--', color='green',lw=lw, label='AT8 testing (AUC {:.2f})'.format(AT8_auc_t))
    plt.plot(AT8_recall_v,AT8_prec_v,  color='green', lw=lw, label='AT8 validation (AUC {:.2f})'.format(AT8_auc_v))
    plt.plot(AT8_x_thres_t, AT8_y_thres_t, color='green', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(AT8_x_thres_v, AT8_y_thres_v, color='green', lw=lw, marker='*', markersize=12)

    plt.plot(MC1_recall_t,MC1_prec_t,'--', color='blue',lw=lw, label='MC1 testing (AUC {:.2f})'.format(MC1_auc_t))
    plt.plot(MC1_recall_v,MC1_prec_v,  color='blue', lw=lw, label='MC1 validation (AUC {:.2f})'.format(MC1_auc_v))
    plt.plot(MC1_x_thres_t, MC1_y_thres_t, color='blue', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(MC1_x_thres_v, MC1_y_thres_v, color='blue', lw=lw, marker='*', markersize=12)


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(fig1_name)


    pass


if __name__ == '__main__':
    main()