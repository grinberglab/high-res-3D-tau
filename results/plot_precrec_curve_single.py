import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx


def main():

    thres = 0.7


    #AT100

    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AT100/results/AT100_precision_recall_all.png'

    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
    AV13_AT100_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results/testing/AV13_AT100_testing_stats.npy')
    AV13_AT100_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT100/results/validation/AV13_AT100_validation_stats.npy')
    AV23_AT100_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT100/results/testing/AV23_AT100_testing_stats.npy')
    AV23_AT100_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT100/results/validation/AV23_AT100_validation_stats.npy')


    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]

    AV13_AT100_prec_t = AV13_AT100_test_stats[:, 2]; AV13_AT100_recall_t = AV13_AT100_test_stats[:, 3]
    AV13_AT100_prec_v = AV13_AT100_val_stats[:, 2]; AV13_AT100_recall_v = AV13_AT100_val_stats[:, 3]
    AV23_AT100_prec_t = AV23_AT100_test_stats[:, 2]; AV23_AT100_recall_t = AV23_AT100_test_stats[:, 3]
    AV23_AT100_prec_v = AV23_AT100_val_stats[:, 2]; AV23_AT100_recall_v = AV23_AT100_val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    plt.figure()
    lw = 2

    plt.plot(AV13_AT100_recall_t,AV13_AT100_prec_t,'--', color='red',lw=lw, label=' #1 AT100 Test')
    plt.plot(AV13_AT100_recall_v,AV13_AT100_prec_v,'--',  color='green', lw=lw, label='#1 AT100 Validation')
    plt.plot(AV23_AT100_recall_t,AV23_AT100_prec_t,'--', color='blue',lw=lw, label=' #2 AT100 Test')
    plt.plot(AV23_AT100_recall_v,AV23_AT100_prec_v,'--',  color='peru', lw=lw, label='#2 AT100 Validation')

    plt.plot(recall_t,prec_t, color='darkorange',lw=lw, label='All Testing')
    plt.plot(recall_v,prec_v,  color='purple', lw=lw, label='All Validation')


    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AT100 Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(fig1_name)
    #
    #
    # #AT8

    fig1_name = '/home/maryana/storage2/Posdoc/AVID/AT8/results/AT8_precision_recall_all.png'

    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/testing/AT8_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_validation_stats.npy')
    AV13_AT8_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results/testing/AV13_AT8_testing_stats.npy')
    AV13_AT8_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/AT8/results/validation/AV13_AT8_validation_stats.npy')
    AV23_AT8_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT8/results/testing/AV23_AT8_testing_stats.npy')
    AV23_AT8_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/AT8/results/validation/AV23_AT8_validation_stats.npy')


    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]

    AV13_AT8_prec_t = AV13_AT8_test_stats[:, 2]; AV13_AT8_recall_t = AV13_AT8_test_stats[:, 3]
    AV13_AT8_prec_v = AV13_AT8_val_stats[:, 2]; AV13_AT8_recall_v = AV13_AT8_val_stats[:, 3]
    AV23_AT8_prec_t = AV23_AT8_test_stats[:, 2]; AV23_AT8_recall_t = AV23_AT8_test_stats[:, 3]
    AV23_AT8_prec_v = AV23_AT8_val_stats[:, 2]; AV23_AT8_recall_v = AV23_AT8_val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    plt.figure()
    lw = 2

    plt.plot(AV13_AT8_recall_t,AV13_AT8_prec_t,'--', color='red',lw=lw, label=' #1 AT8 Test')
    plt.plot(AV13_AT8_recall_v,AV13_AT8_prec_v,'--',  color='green', lw=lw, label='#1 AT8 Validation')
    plt.plot(AV23_AT8_recall_t,AV23_AT8_prec_t,'--', color='blue',lw=lw, label=' #2 AT8 Test')
    plt.plot(AV23_AT8_recall_v,AV23_AT8_prec_v,'--',  color='peru', lw=lw, label='#2 AT8 Validation')

    plt.plot(recall_t,prec_t, color='darkorange',lw=lw, label='All Testing')
    plt.plot(recall_v,prec_v,  color='purple', lw=lw, label='All Validation')


    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AT8 Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(fig1_name)

    # MC1

    fig1_name = '/home/maryana/storage2/Posdoc/AVID/MC1/results/MC1_precision_recall_all.png'

    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')
    AV13_MC1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results/testing/AV13_MC1_testing_stats.npy')
    AV13_MC1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV13/MC1/results/validation/AV13_MC1_validation_stats.npy')
    AV23_MC1_test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/MC1/results/testing/AV23_MC1_testing_stats.npy')
    AV23_MC1_val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AV23/MC1/results/validation/AV23_MC1_validation_stats.npy')


    prec_t = test_stats[:, 2]; recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]; recall_v = val_stats[:, 3]

    AV13_MC1_prec_t = AV13_MC1_test_stats[:, 2];
    AV13_MC1_recall_t = AV13_MC1_test_stats[:, 3]
    AV13_MC1_prec_v = AV13_MC1_val_stats[:, 2];
    AV13_MC1_recall_v = AV13_MC1_val_stats[:, 3]

    AV23_MC1_prec_t = AV23_MC1_test_stats[:, 2];
    AV23_MC1_recall_t = AV23_MC1_test_stats[:, 3]
    AV23_MC1_prec_v = AV23_MC1_val_stats[:, 2];
    AV23_MC1_recall_v = AV23_MC1_val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    plt.figure()
    lw = 2

    plt.plot(AV13_MC1_recall_t,AV13_MC1_prec_t,'--', color='red',lw=lw, label=' #1 MC1 Test')
    plt.plot(AV13_MC1_recall_v,AV13_MC1_prec_v,'--',  color='green', lw=lw, label='#1 MC1 Validation')
    plt.plot(AV23_MC1_recall_t,AV23_MC1_prec_t,'--', color='blue',lw=lw, label=' #2 MC1 Test')
    plt.plot(AV23_MC1_recall_v,AV23_MC1_prec_v,'--',  color='peru', lw=lw, label='#2 MC1 Validation')

    plt.plot(recall_t,prec_t, color='darkorange',lw=lw, label='All Testing')
    plt.plot(recall_v,prec_v,  color='purple', lw=lw, label='All Validation')


    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('MC1 Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.savefig(fig1_name)


if __name__ == '__main__':
    main()