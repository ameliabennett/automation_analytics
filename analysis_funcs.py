import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

def recall_at_depth(y_true, y_pred_proba, depth):
    assert depth == float(depth)
    assert depth <= 1.0
    assert depth >= 0.0
    assert len(y_true) == len(y_pred_proba)
    y_array = np.vstack((y_true, y_pred_proba))
    y_array = np.flip(y_array[:,y_array[1,:].argsort()])
    length_by_depth = int(round(len(y_true) * depth))
    return sum(y_array[1,0:length_by_depth])/sum(y_true)

def depth_at_recall(y_true, y_pred_proba, recall):
    assert recall == float(depth)
    assert recall <= 1.0
    assert recall >= 0.0
    assert len(y_true) == len(y_pred_proba)
    y_array = np.vstack((y_true, y_pred_proba))
    y_array = np.flip(y_array[:,y_array[1,:].argsort()])
    recall_sum = int(round(sum(y_true) * recall))
    length = len(y_true)
    cum_sum = np.cumsum(y_array[1])
    #first_cum = np.nonzero(cum_sum == recall_sum)[0]
    first_cum = list(cum_sum == recall_sum).index(1)
    return first_cum / length

def precision_at_depth(y_true, y_pred_proba, depth):
    assert depth == float(depth)
    assert depth <= 1.0
    assert depth >= 0.0
    assert len(y_true) == len(y_pred_proba)
    y_array = np.vstack((y_true, y_pred_proba))
    y_array = np.flip(y_array[:,y_array[1,:].argsort()])
    length_by_depth = int(round(len(y_true) * depth))
    return sum(y_array[1,0:length_by_depth])/length_by_depth

def output_function(y_true, y_pred_proba):
    assert len(y_true) == len(y_pred_proba)
    y_array = np.vstack((y_true, y_pred_proba))
    y_array = np.flip(y_array[:,y_array[1,:].argsort()])
    def recall_at_depth_internal(y_array, depth):
        return sum(y_array[1:0:int(round(len(y_array[0]) * depth))])/sum(y_array[1])
    def precision_at_depth_internal(y_array, depth):
        return sum(y_array[1:0:int(round(len(y_array[0]) * depth))])/int(round(len(y_array[0]) * depth))
    def tp_at_depth_internal(y_array, depth):
        return sum(y_array[1:0:int(round(len(y_array[0]) * depth))])
    def fp_at_depth_internal(y_array, depth):
        return int(round(len(y_array[0]) * depth)) - sum(y_array[1:0:int(round(len(y_array[0]) * depth))])
    def pred_proba_at_depth(y_array, depth):
        return y_array[0, int(round(len(y_array[0]) * depth))-1]
    depth_array = np.linspace(0.05, 1, 20)
    depth_array = np.sort(depth_array)
    cum_recall_list = list()
    cum_precision_list = list()
    cum_count_list = list()
    cum_tp_list = list()
    cum_fp_list = list()
    cum_pred_proba_list = list()
    for elem in depth_array:
        cum_recall_list.append(recall_at_depth_internal(y_array, elem))
        cum_precision_list.append(precision_at_depth_internal(y_array, elem))
        cum_count_list.append(int(round(len(y_array[0]) * elem)))
        cum_tp_list.append(tp_at_depth_internal(y_array, elem))
        cum_fp_list.append(fp_at_depth_internal(y_array, elem))
        cum_pred_proba_list(pred_proba_at_depth(y_array, elem))
    output_array = np.stack((depth_array, np.asarray(cum_pred_proba_list), np.asarray(cum_recall_list),
                             np.asarray(cum_precision_list), np.asarray(cum_count_list),
                             np.asarray(cum_tp_list), np.asarray(cum_fp_list)), axis = 1)
    output_df = pd.DataFrame(output_array)
    output_df.columns = ['Depth','Probability Threshold','Recall','Precision','Count','True Positives','False Positives']
    return output_df

def plot_binary_auc(y_true, y_pred_proba):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_proba)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color = 'navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def thresholder(y_test, y_pred_proba, recall_requirement):
    max_depth = depth_at_recall(y_test, y_pred_proba, recall_requirement).min()
    decision_point = int(max_depth*len(y_test))
    sorted_proba = sorted(y_pred_proba)
    return sorted_proba[-decision_point]