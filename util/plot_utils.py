import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


'''
------------------------------------------------------------------------------------------
Result plots:
'''
def various_metrics(df_res, eps = 1e-6):
    tn, fp, fn, tp = confusion_matrix(df_res['ground_truth'], df_res['prediction'], labels = [0, 1]).ravel()
    acc = (tn+tp) / (tn+fp+tp+fn + eps)
    prec = tp/(tp+fp + eps)
    recall = tp/(tp+fn + eps)
    f2 = 2*(prec*recall)/(prec+recall + eps)
    specificity = tn / (tn+fp + eps)
    npv = tn/(tn+fn + eps)
    return acc, prec, recall, f2, specificity, npv

def obtain_list_of_metrics(df_res, n_conf):

    confidence = []
    percs_subset = []
    accuracies = []
    precisions = []
    recalls = []
    f2s = []
    specificities = []
    npvs = []


    for diff in np.linspace(0, 0.49, n_conf):
        subset = df_res[(df_res['probability'] > 0.5+diff)|(df_res['probability'] < 0.5-diff)]
        N_subset = subset.shape[0]
        perc_N = N_subset/df_res.shape[0]

        confidence.append(0.5+diff)
        percs_subset.append(perc_N)

        acc, prec, recall, f2, specificity, npv = various_metrics(subset)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(recall)
        f2s.append(f2)
        specificities.append(specificity)
        npvs.append(npv)
        
    return confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs

def plot_results(confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs, title, suptitle):
    merged_x_axis = []
    for i in range(0, len(percs_subset)):
        tuple_to_print = (np.round(confidence[i],2), np.round(percs_subset[i], 2))
        merged_x_axis.append('\n\n'.join(str(x) for x in tuple_to_print))

    figure(figsize=(8, 6), dpi=80)

    plt.plot(merged_x_axis, accuracies, label = 'accuracy', linewidth=2)
    plt.plot(merged_x_axis, recalls, label = 'recall', linestyle = '--')
    plt.plot(merged_x_axis, precisions, label = 'precision', linestyle = '--')
    plt.plot(merged_x_axis, specificities, label = 'specificity', linestyle = '-.')
    plt.plot(merged_x_axis, npvs, label = 'npv', linestyle = ':')
    plt.title(title)
    if len(suptitle)>0:
        plt.suptitle(suptitle)
    plt.xlabel(f"Confidence & Perc%")
    plt.legend()
    plt.show()
    
def obtain_plot(df_res, n_original_df = np.nan, title = 'Metrics', n_conf = 10):
    confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs = obtain_list_of_metrics(df_res, n_conf)

    perc = np.round(df_res.shape[0]/n_original_df*100, 2)

    vc = df_res.ground_truth.value_counts()
    count0 = vc.loc[0] if 0 in vc.index else 0
    count1 = vc.loc[1] if 1 in vc.index else 0
    perc_0, perc_1 = np.round(count0/df_res.shape[0], 2), np.round(count1/df_res.shape[0], 2)
    
    plot_results(confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs, title = title + f'({perc}% of the data)', suptitle = f'0% :{perc_0}, 1% :{perc_1}')
    
    
'''
------------------------------------------------------------------------------------------
Log plots:
'''

def plot_logs(log, metric, best_model):
    assert metric in ['loss', 'accuracy', 'precision', 'recall', 'f2']
    
    mapping = {'loss': 'loss_ce', 'accuracy':'accuracy', 
               'precision':'precision', 'recall':'recall', 'f2':'F2'}
    
    training_color = '#104E8B'
    val_color = '#EE9572'
    
    plt.plot(log['epoch'], log[f'train_{mapping[metric]}'], label = 'training', color = training_color)
    plt.plot(log['epoch'], log[f'test_{mapping[metric]}'], label = 'val', color = val_color)
    plt.title(metric)
    plt.xlabel("epochs")
    plt.axvline(x = best_model, color = 'gray', label = 'best model', linestyle = '--')
    plt.legend()
    plt.show()

'''
------------------------------------------------------------------------------------------
'''
