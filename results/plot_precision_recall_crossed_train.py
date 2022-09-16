import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx


def main():

    thres = 0.5

    #AT100 segmented using model trained with AT8
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
    cross_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results_trained_with_AT8/testing/AT100_AT8_testing_stats.npy')
    cross_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results_trained_with_AT8/validation/AT100_AT8_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AT100/results_trained_with_AT8/AT100_AT8_prec_recall.png'

    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]

    fpr_t = test_stats[:,5]; f1_t = test_stats[:,4]
    fpr_v = val_stats[:,5]; f1_v = val_stats[:,4]


    cross_prec_t = cross_test_stats[:, 2]; cross_recall_t = cross_test_stats[:, 3]
    cross_prec_v = cross_val_stats[:, 2]; cross_recall_v = cross_val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    print('AT100')
    print('Testing: {}(Prec) {}(Rec) {}(FPR) {}(F1)'.format(prec_t[index],recall_t[index],fpr_t[index],f1_t[index]))
    print('Val: {}(Prec) {}(Rec) {}(FPR) {}(F1)'.format(prec_v[index],recall_v[index],fpr_v[index],f1_v[index]))

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    plt.figure()
    lw = 2

    plt.plot(recall_t,prec_t, color='darkorange',lw=lw, label='Full testing')
    plt.plot(recall_v,prec_v,  color='purple', lw=lw, label='Full validation')
    plt.plot(cross_recall_t,cross_prec_t,'--', color='red',lw=lw, label='AT100/AT8 testing')
    plt.plot(cross_recall_v,cross_prec_v,'--',  color='blue', lw=lw, label='AT100/AT8 validation')

    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AT100/AT8 cross-testing precision-recall curve')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)


    #AT8 segmented using model trained with AT100
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/testing/AT8_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_validation_stats.npy')
    cross_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results_trained_with_AT100/testing/AT8_AT100_testing_stats.npy')
    cross_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results_trained_with_AT100/validation/AT8_AT100_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AT8/results_trained_with_AT100/AT8_AT100_prec_recall.png'

    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]

    fpr_t = test_stats[:,5]; f1_t = test_stats[:,4]
    fpr_v = val_stats[:,5]; f1_v = val_stats[:,4]

    cross_prec_t = cross_test_stats[:, 2]; cross_recall_t = cross_test_stats[:, 3]
    cross_prec_v = cross_val_stats[:, 2]; cross_recall_v = cross_val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    print('AT8')
    print('Testing: {}(Prec) {}(Rec) {}(FPR) {}(F1)'.format(prec_t[index],recall_t[index],fpr_t[index],f1_t[index]))
    print('Val: {}(Prec) {}(Rec) {}(FPR) {}(F1)'.format(prec_v[index],recall_v[index],fpr_v[index],f1_v[index]))

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    plt.figure()
    lw = 2


    plt.plot(recall_t,prec_t, color='darkorange',lw=lw, label='Full testing')
    plt.plot(recall_v,prec_v,  color='purple', lw=lw, label='Full validation')
    plt.plot(cross_recall_t,cross_prec_t,'--', color='red',lw=lw, label='AT8/AT100 testing')
    plt.plot(cross_recall_v,cross_prec_v,'--',  color='blue', lw=lw, label='AT8/AT100 validation')


    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AT8/AT100 cross-testing precision-recall curve')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)

    #MC1 segmented using model trained with AT100
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')
    cross_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results_trained_with_AT100/testing/MC1_AT100_testing_stats.npy')
    cross_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results_trained_with_AT100/validation/MC1_AT100_validation_stats.npy')
    fig1_name = '/home/maryana/storage2/Posdoc/AVID/MC1/results_trained_with_AT100/MC1_AT100_prec_recall.png'

    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]

    fpr_t = test_stats[:,5]; f1_t = test_stats[:,4]
    fpr_v = val_stats[:,5]; f1_v = val_stats[:,4]

    cross_prec_t = cross_test_stats[:, 2]; cross_recall_t = cross_test_stats[:, 3]
    cross_prec_v = cross_val_stats[:, 2]; cross_recall_v = cross_val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    print('MC1')
    print('Testing: {}(Prec) {}(TPR) {}(FPR) {}(F1) {}(TNR) {}(FNR)'.format(prec_t[index],recall_t[index],fpr_t[index],f1_t[index],1-fpr_t[index],1-recall_t[index]))
    print('Val: {}(Prec) {}(TPR) {}(FPR) {}(F1) {}(TNR) {}(FNR)'.format(prec_v[index],recall_v[index],fpr_v[index],f1_v[index],1-fpr_v[index],1-recall_v[index]))

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    plt.figure()
    lw = 2


    plt.plot(recall_t,prec_t, color='darkorange',lw=lw, label='Full testing')
    plt.plot(recall_v,prec_v,  color='purple', lw=lw, label='Full validation')
    plt.plot(cross_recall_t,cross_prec_t,'--', color='red',lw=lw, label='MC1/AT100 testing')
    plt.plot(cross_recall_v,cross_prec_v,'--',  color='blue', lw=lw, label='MC1/AT100 Validation')


    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('MC1/AT100 cross-testing precision-recall curve')
    #plt.legend(loc="lower right")
    #plt.show()

    #plt.savefig(fig1_name)

    pass


if __name__ == '__main__':
    main()