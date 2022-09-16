import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

def find_nearest1(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx


def main():

    thres = 0.5


    # #AT100
     test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/testing/AT100_testing_stats.npy')
     val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_validation_stats.npy')
     fig1_name = '/home/maryana/storage2/Posdoc/AVID/AT100/results/validation/AT100_precision_recall.png'

     prec_t = test_stats[:, 2]
     recall_t = test_stats[:, 3]
     prec_v = val_stats[:, 2]
     recall_v = val_stats[:, 3]

     probs = np.linspace(1, 0, num=20)
     index = find_nearest1(probs,thres)

     x_thres_t = recall_t[index]
     y_thres_t = prec_t[index]
     x_thres_v = recall_v[index]
     y_thres_v = prec_v[index]

     auc_t = auc(recall_t, prec_t)
     auc_v = auc(recall_v, prec_v)

     plt.figure()
     lw = 2
     plt.plot(recall_t,prec_t, color='darkorange',lw=lw, label='Testing')
     plt.plot(recall_v,prec_v,  color='purple', lw=lw, label='Validation')
     plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
     plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
     plt.xlim([0.0, 1.0])
     plt.ylim([0.0, 1.05])
     plt.xlabel('Recall')
     plt.ylabel('Precision')
     plt.title('AT100 Precision-Recall Curve')
     plt.legend(loc="lower right")
     plt.show()

    # plt.savefig(fig1_name)
    #
    #
    # #AT8
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/testing/AT8_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_validation_stats.npy')
    fig2_name = '/home/maryana/storage2/Posdoc/AVID/AT8/results/validation/AT8_precision_recall.png'

    prec_t = test_stats[:, 2]
    recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]
    recall_v = val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs,thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    auc_t = auc(recall_t, prec_t)
    auc_v = auc(recall_v, prec_v)
    #
    plt.figure()
    lw = 2
    plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing (AUC {:.2f})'.format(auc_t))
    plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation (AUC {:.2f})'.format(auc_v))
    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12) # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AT8 Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()
    #
    # # plt.savefig(fig2_name)
    #
    #
    # # MC1
    # test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/testing/MC1_testing_stats.npy')
    # val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_validation_stats.npy')
    # fig2_name = '/home/maryana/storage2/Posdoc/AVID/MC1/results/validation/MC1_precision_recall.png'
    #
    # prec_t = test_stats[:, 2]
    # recall_t = test_stats[:, 3]
    # prec_v = val_stats[:, 2]
    # recall_v = val_stats[:, 3]
    #
    # probs = np.linspace(1, 0, num=20)
    # index = find_nearest1(probs, thres)
    #
    # x_thres_t = recall_t[index]
    # y_thres_t = prec_t[index]
    # x_thres_v = recall_v[index]
    # y_thres_v = prec_v[index]
    #
    # auc_t = auc(recall_t, prec_t)
    # auc_v = auc(recall_v, prec_v)
    #
    # plt.figure()
    # lw = 2
    # plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing (AUC {:.2f})'.format(auc_t))
    # plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation (AUC {:.2f})'.format(auc_v))
    # plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*',
    #          markersize=12)  # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    # plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('MC1 Precision-Recall Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    #
    # # plt.savefig(fig2_name)
    #
    # EoL with all AVID
    # test_stats = np.load('/home/maryana/storage2/Posdoc/End_of_Life/result/testing/End_of_Life_testing_stats.py.npy')
    # val_stats = np.load('/home/maryana/storage2/Posdoc/End_of_Life/result/validation/End_of_Life_validation_stats.py.npy')
    # fig2_name = '/home/maryana/storage2/Posdoc/End_of_Life/result/EoL_precision_recall.png'
    #
    # prec_t = test_stats[:, 2]
    # recall_t = test_stats[:, 3]
    # prec_v = val_stats[:, 2]
    # recall_v = val_stats[:, 3]
    #
    # probs = np.linspace(1, 0, num=20)
    # index = find_nearest1(probs, thres)
    #
    # x_thres_t = recall_t[index]
    # y_thres_t = prec_t[index]
    # x_thres_v = recall_v[index]
    # y_thres_v = prec_v[index]
    #
    # auc_t = auc(recall_t, prec_t)
    # auc_v = auc(recall_v, prec_v)
    #
    # plt.figure()
    # lw = 2
    # plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing')
    # plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation')
    # plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*',
    #          markersize=12)  # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    # plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('EoL Precision-Recall Curve')
    # plt.legend(loc="lower right")
    # plt.show()

    # # EoL with AT8
    test_stats = np.load('/home/maryana/storage2/Posdoc/EoL+AT8_new/result/AT8+EoL_testing_stat.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/EoL+AT8_new/result/AT8+EoL_validation_stat.npy')
    fig2_name = '/home/maryana/storage2/Posdoc/EoL+AT8/result/EoL_AT8_precision_recall.png'

    prec_t = test_stats[:, 2]
    recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]
    recall_v = val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs, thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    auc_t = auc(recall_t, prec_t)
    auc_v = auc(recall_v, prec_v)

    plt.figure()
    lw = 2
    plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing (AUC {:.2f})'.format(auc_t))
    plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation (AUC {:.2f})'.format(auc_v))
    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12)  # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('EoL with AT8 new Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    # EoL only
    # test_stats = np.load('/home/maryana/storage/Posdoc/EoL_only/result/testing/heat_map/EoL_only_testing_stat.npy')
    # val_stats = np.load('/home/maryana/storage/Posdoc/EoL_only/result/validation/heat_map/EoL_only_validation_stat.npy')
    # fig2_name = '/home/maryana/storage/Posdoc/EoL_only/result/EoL_only_precision_recall.png'

    # prec_t = test_stats[:, 2]
    # recall_t = test_stats[:, 3]
    # prec_v = val_stats[:, 2]
    # recall_v = val_stats[:, 3]
    #
    # probs = np.linspace(1, 0, num=20)
    # index = find_nearest1(probs, thres)
    #
    # x_thres_t = recall_t[index]
    # y_thres_t = prec_t[index]
    # x_thres_v = recall_v[index]
    # y_thres_v = prec_v[index]
    #
    # auc_t = auc(recall_t, prec_t)
    # auc_v = auc(recall_v, prec_v)
    #
    # plt.figure()
    # lw = 2
    # plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing (AUC {:.2f})'.format(auc_t))
    # plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation (AUC {:.2f})'.format(auc_v))
    # plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*',
    #          markersize=12)  # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    # plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('EoL Precision-Recall Curve')
    # plt.legend(loc="lower right")
    # plt.show()

    # EoL with AT8 new deblurred
    #test_stats = np.load('/home/maryana/storage2/Posdoc/EoL+AT8_new_Deblurred_stat/db_testing/EoL+AT8_new_Deblurred_testing_stats.npy')
    #val_stats = np.load('/home/maryana/storage2/Posdoc/EoL+AT8_new_Deblurred_stat/db_validation/EoL+AT8_new_Deblurred_validation_stats.npy')
    #fig2_name = '/home/maryana/storage2/Posdoc/EoL+AT8_new_Deblurred_stat/AT8+EoL_new_precision_recall.png'

    #prec_t = test_stats[:, 2]
    #recall_t = test_stats[:, 3]
    #prec_v = val_stats[:, 2]
    #recall_v = val_stats[:, 3]

    #probs = np.linspace(1, 0, num=20)
    #index = find_nearest1(probs, thres)

    #x_thres_t = recall_t[index]
    #y_thres_t = prec_t[index]
    #x_thres_v = recall_v[index]
    #y_thres_v = prec_v[index]

    #auc_t = auc(recall_t, prec_t)
    #auc_v = auc(recall_v, prec_v)

    #plt.figure()
    #lw = 2
    #plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing (AUC {:.2f})'.format(auc_t))
    #plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation (AUC {:.2f})'.format(auc_v))
    #plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12)  # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    #plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.title('EoL Precision-Recall Curve')
    #plt.legend(loc="lower right")
    #plt.show()

    # EoL_CP13+MAOB+AV_AT8_sgd_200_epoch
    test_stats = np.load('/home/maryana/storage2/Posdoc/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch/result/testing/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch_testing_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch/result/validation/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch_validation_stats.npy')
    fig5_name = '/home/maryana/storage2/Posdoc/EoL+AT8/result/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch.png'

    prec_t = test_stats[:, 2]
    recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]
    recall_v = val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs, thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    auc_t = auc(recall_t, prec_t)
    auc_v = auc(recall_v, prec_v)

    plt.figure()
    lw = 2
    plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing (AUC {:.2f})'.format(auc_t))
    plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation (AUC {:.2f})'.format(auc_v))
    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12)  # Testing threshold taken from prec/recall vectors using the index of probs closest to the threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('EoL_CP13+MAOB+AV_AT8_sgd_200_epoch Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    # AT8_combined_on_EoL_CP13+MAOB+AV_AT8_sgd_200_epoch
    test_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/EOL_testing/AT8_Testing_EOL_stats.npy')
    val_stats = np.load('/home/maryana/storage2/Posdoc/AVID/AT8/results/EOL_validation/AT8_Validation_EOL_stats.npy')
    fig6_name = '/home/maryana/storage2/Posdoc/EoL+AT8/result/EoL_CP13+MAOB+AV_AT8_sgd_200_epoch.png'

    prec_t = test_stats[:, 2]
    recall_t = test_stats[:, 3]
    prec_v = val_stats[:, 2]
    recall_v = val_stats[:, 3]

    probs = np.linspace(1, 0, num=20)
    index = find_nearest1(probs, thres)

    x_thres_t = recall_t[index]
    y_thres_t = prec_t[index]
    x_thres_v = recall_v[index]
    y_thres_v = prec_v[index]

    auc_t = auc(recall_t, prec_t)
    auc_v = auc(recall_v, prec_v)

    plt.figure()
    lw = 2
    plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing (AUC {:.2f})'.format(auc_t))
    plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation (AUC {:.2f})'.format(auc_v))
    plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*', markersize=12)  # Testing threshold taken from prec/recall vectors using the index of probs closest to the threshold = 0.7
    plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AT8_combined_on_EoL_CP13+MAOB+AV_AT8_sgd_200_epoch_PRC')
    plt.legend(loc="lower right")
    plt.show()

    # # EoL with AT100 new
    # test_stats = np.load('/home/maryana/storage/Posdoc/EoL+AT100_orig_stat/db_testing/EoL+AT100_orig_testing.npy')
    # val_stats = np.load('/home/maryana/storage/Posdoc/EoL+AT100_orig_stat/db_validation/EoL+AT100_orig_validation.npy')
    # fig2_name = '/home/maryana/storage2/Posdoc/EoL+AT8_new_Deblurred_stat/AT8+EoL_new_precision_recall.png'
    #
    # prec_t = test_stats[:, 2]
    # recall_t = test_stats[:, 3]
    # prec_v = val_stats[:, 2]
    # recall_v = val_stats[:, 3]
    #
    # probs = np.linspace(1, 0, num=20)
    # index = find_nearest1(probs, thres)
    #
    # x_thres_t = recall_t[index]
    # y_thres_t = prec_t[index]
    # x_thres_v = recall_v[index]
    # y_thres_v = prec_v[index]
    #
    # auc_t = auc(recall_t, prec_t)
    # auc_v = auc(recall_v, prec_v)
    #
    # plt.figure()
    # lw = 2
    # plt.plot(recall_t, prec_t, color='darkorange', lw=lw, label='Testing (AUC {:.2f})'.format(auc_t))
    # plt.plot(recall_v, prec_v, color='purple', lw=lw, label='Validation (AUC {:.2f})'.format(auc_v))
    # plt.plot(x_thres_t, y_thres_t, color='red', lw=lw, marker='*',
    #          markersize=12)  # Testing threshold tirado dos vetores prec/recall usando o index de probs mais proximos do threshold = 0.7
    # plt.plot(x_thres_v, y_thres_v, color='red', lw=lw, marker='*', markersize=12)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('EoL Precision-Recall Curve')
    # plt.legend(loc="lower right")
    # plt.show()


    pass


if __name__ == '__main__':
    main()