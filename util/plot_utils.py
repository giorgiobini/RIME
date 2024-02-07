import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve, auc


'''
------------------------------------------------------------------------------------------
Result plots:
'''

def plot_roc_curves(models, ground_truth):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')  # Plotting the random guessing line

    for model in models:
        probabilities = model['prob']
        model_name = model['model_name']

        fpr, tpr, _ = roc_curve(ground_truth, probabilities)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.show()

def various_metrics(df_res, eps = 1e-6):
    tn, fp, fn, tp = confusion_matrix(df_res['ground_truth'], df_res['prediction'], labels = [0, 1]).ravel()
    acc = (tn+tp) / (tn+fp+tp+fn + eps)
    prec = tp/(tp+fp + eps)
    recall = tp/(tp+fn + eps)
    f2 = 2*(prec*recall)/(prec+recall + eps)
    specificity = tn / (tn+fp + eps)
    npv = tn/(tn+fn + eps)
    return acc, prec, recall, f2, specificity, npv

def remove_last_n_if_below_threshold(confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs, excluding_treshold):
    N = 0
    for i in range(len(percs_subset) - 1, -1, -1):
        if percs_subset[i] < excluding_treshold:
            N += 1
        else:
            break
    
    if N > 0:
        for lst in [confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs]:
            lst[:] = lst[:-N]

def obtain_list_of_metrics(df_res, n_conf, excluding_treshold, ensemble = False):

    confidence = []
    percs_subset = []
    accuracies = []
    precisions = []
    recalls = []
    f2s = []
    specificities = []
    npvs = []

    for diff in np.linspace(0, 0.49, n_conf): #0.49
        if ensemble:
            subset = df_res[(df_res['ensemble_score'] > 0.5+diff)|(df_res['ensemble_score'] < 0.5-diff)]
        else:
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
        
    remove_last_n_if_below_threshold(confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs, excluding_treshold)
    
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
    
def obtain_plot(df_res, n_original_df = np.nan, title = 'Metrics', n_conf = 10, excluding_treshold = 0.01, plot_perc = False, ensemble = False):
    confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs = obtain_list_of_metrics(df_res, n_conf, excluding_treshold, ensemble = ensemble)

    perc = np.round(df_res.shape[0]/n_original_df*100, 2)

    vc = df_res.ground_truth.value_counts()
    count0 = vc.loc[0] if 0 in vc.index else 0
    count1 = vc.loc[1] if 1 in vc.index else 0
    perc_0, perc_1 = np.round(count0/df_res.shape[0], 2), np.round(count1/df_res.shape[0], 2)
    if plot_perc:
        plot_results(confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs, title = title + f'({perc}% of the data)', suptitle = f'0% :{perc_0}, 1% :{perc_1}')
    else:
        plot_results(confidence, percs_subset, accuracies, precisions, recalls, f2s, specificities, npvs, title = title, suptitle = '')
    
'''
------------------------------------------------------------------------------------------
Log plots:
'''

def plot_logs(log, metric, best_model):

    training_color = '#104E8B'
    val_color = '#EE9572'
    
    plt.plot(log['epoch'], log[f'train_{metric}'], label = 'training', color = training_color)
    plt.plot(log['epoch'], log[f'test_{metric}'], label = 'val', color = val_color)
    plt.title(metric)
    plt.xlabel("epochs")
    plt.axvline(x = best_model, color = 'gray', label = 'best model', linestyle = '--')
    plt.legend()
    plt.show()

'''
------------------------------------------------------------------------------------------
'''

def balance_df(df, n_iter = 25):
    toappend = []
    if df[df.ground_truth == 0].shape[0] > df[df.ground_truth == 1].shape[0]:
        for i in range(n_iter):
            negs = df[df.ground_truth == 0]
            poss = df[df.ground_truth == 1]
            toappend.append(pd.concat([negs.sample(len(poss)), poss], axis = 0))
    else:
        for i in range(n_iter):
            negs = df[df.ground_truth == 0]
            poss = df[df.ground_truth == 1]
            toappend.append(pd.concat([poss.sample(len(negs)), negs], axis = 0))
    balanced = pd.concat(toappend, axis = 0)
    return balanced

def collect_results_based_on_confidence_level_based_on_treshold(df, how = 'intarna', n_values = 15, balance = True, MIN_PERC = 0.05):
    auc_nt = []
    auc_intarna = []
    merged_x_axis = []
    
    n_total = df.shape[0]
    
    confidence_space = np.linspace(0.51, 0.99, n_values)
    for i in range(n_values):
        
        if how == 'intarna':
            treshold = 1-confidence_space[i]
            subset = df[
                (df.E_norm <= df.E_norm.quantile(treshold))|
                (df.E_norm >= df.E_norm.quantile(1-treshold))
            ]
        elif how == 'nt':
            treshold = confidence_space[i]
            subset = df[
                (df.probability >= treshold)|
                (df.probability <= (1-treshold))
            ]
        else:
            raise NotImplementedError
        
        n_subset = subset.shape[0]
        if balance:
            subset = balance_df(subset)
        
        if n_subset/n_total > MIN_PERC:
            fpr, tpr, _ = roc_curve(subset.ground_truth, subset.probability)
            roc_auc = auc(fpr, tpr)
            auc_nt.append(roc_auc)

            fpr, tpr, _ = roc_curve(abs(1 - subset.ground_truth), subset.E_norm)
            roc_auc = auc(fpr, tpr)
            auc_intarna.append(roc_auc)
            
            tuple_to_print = (np.round(confidence_space[i],2), np.round(subset.shape[0]/n_total, 2))
            
            merged_x_axis.append('\n\n'.join(str(x) for x in tuple_to_print))
        
    return merged_x_axis, auc_nt, auc_intarna


def collect_results_based_on_confidence_level_based_on_percentile(df, how = 'intarna', n_values = 15, balance = True, MIN_PERC = 1, space = 'linear', calc_ens = True):
    
    total_data = df.shape[0]
    
    if space == 'linear':
        percs_data = np.linspace(MIN_PERC, 100, n_values)[::-1]
    elif space == 'log':
        percs_data = np.logspace(np.log10(MIN_PERC), np.log10(100), n_values)[::-1]
        
    auc_nt = []
    auc_intarna = []
    auc_ens = []
    x_axis = []
    
    n_total = df.shape[0]

    for i in range(n_values):
        
        n_to_sample = int(math.ceil(percs_data[i]/100 * total_data))

        if how == 'intarna':
            subset = pd.concat([
                df.sort_values('E_norm').head(math.ceil(n_to_sample/2)), 
                df.sort_values('E_norm').tail(math.ceil(n_to_sample/2))
            ], axis = 0)
            
        elif how == 'nt':
            subset = pd.concat([
                df.sort_values('probability').head(math.ceil(n_to_sample/2)), 
                df.sort_values('probability').tail(math.ceil(n_to_sample/2))
            ], axis = 0)
            
        elif how == 'ensemble':
            subset = pd.concat([
                df.sort_values('ensemble_score').head(math.ceil(n_to_sample/2)), 
                df.sort_values('ensemble_score').tail(math.ceil(n_to_sample/2))
            ], axis = 0)
        else:
            raise NotImplementedError
        
        n_subset = subset.shape[0]
        
        if balance:
            subset = balance_df(subset)
        
        assert n_subset/n_total >= MIN_PERC/100
        assert abs(n_subset/n_total - percs_data[i]/100) < 0.02
        
        
        fpr, tpr, _ = roc_curve(subset.ground_truth, subset.probability)
        roc_auc = auc(fpr, tpr)
        auc_nt.append(roc_auc)

        fpr, tpr, _ = roc_curve(abs(1 - subset.ground_truth), subset.E_norm)
        roc_auc = auc(fpr, tpr)
        auc_intarna.append(roc_auc)
        
        if calc_ens:
            fpr, tpr, _ = roc_curve(subset.ground_truth, subset.ensemble_score)
            roc_auc = auc(fpr, tpr)
            auc_ens.append(roc_auc)
        
        value = str(np.round(percs_data[i], 2))
        x_axis.append(value)
        
    if calc_ens:
        return x_axis, auc_nt, auc_intarna, auc_ens
    else:
        return x_axis, auc_nt, auc_intarna

def collect_results_based_on_confidence_level_how_many1(df, n_values = 15, how = 'nt', balance = False, MIN_PERC = 1, space = 'linear', intarna_treshold = -1.25):
    
    total_data = df.shape[0]
        
    confidence_space = np.linspace(0.51, 0.99, n_values)
        
    perc_1 = []
    x_axis = []
    
    n_total = df.shape[0]

    for i in range(n_values):

        if how == 'intarna':
            treshold = confidence_space[i]
            subset = df[
                (df.E_norm_conf >= treshold)|
                (df.E_norm_conf <= (1-treshold))
            ]
        elif how == 'nt':
            treshold = confidence_space[i]
            subset = df[
                (df.probability >= treshold)|
                (df.probability <= (1-treshold))
            ]
            
        elif how == 'ensemble':
            treshold = confidence_space[i]
            subset = df[
                (df.ensemble_score >= treshold)|
                (df.ensemble_score <= (1-treshold))
            ]
        else:
            raise NotImplementedError
        
        n_subset = subset.shape[0]
        
        if balance:
            subset = balance_df(subset)
        
        if (n_subset/n_total > MIN_PERC/100):

            if how == 'intarna':
                perc_1.append(np.round( (subset.E_norm_conf > 0.5).sum()/subset.shape[0], 2))
            elif how == 'nt':
                perc_1.append(np.round( (subset.prediction == 1).sum()/subset.shape[0], 2))
            elif how == 'ensemble':
                perc_1.append(np.round( (subset.ensemble_score>0.5).sum()/subset.shape[0], 2))

            value = str(np.round(np.round(confidence_space[i],2), 2))

            x_axis.append(value)

    return x_axis, perc_1

def acc_for_each_class(df):
    policies = list(df.policy.value_counts().index)
    for p in policies:
        subset = df[df.policy == p]
        if p == 'easypos':
            acc = np.round(subset[subset.prediction == 1].shape[0]/subset.shape[0], 2)
        else:
            acc = np.round(subset[subset.prediction == 0].shape[0]/subset.shape[0], 2)
        print(f'acc {p}: {acc}')

def calc_acc_for_each_class(df):
    for p in ['easypos', 'smartneg', 'easyneg', 'hardneg']:
        subset = df[df.policy == p]
        if p == 'easypos':
            if subset.shape[0] > 0:
                acc_ep = np.round(subset[subset.prediction == 1].shape[0]/subset.shape[0], 2)
            else: 
                acc_ep = np.nan
        elif p == 'smartneg':
            if subset.shape[0] > 0:
                acc_sn = np.round(subset[subset.prediction == 0].shape[0]/subset.shape[0], 2)
            else: 
                acc_sn = np.nan
        elif p == 'easyneg':
            if subset.shape[0] > 0:
                acc_en = np.round(subset[subset.prediction == 0].shape[0]/subset.shape[0], 2)
            else: 
                acc_en = np.nan
        elif p == 'hardneg':
            if subset.shape[0] > 0:
                acc_hn = np.round(subset[subset.prediction == 0].shape[0]/subset.shape[0], 2)
            else: 
                acc_hn = np.nan
    return acc_ep, acc_sn, acc_en, acc_hn
    
def plot_confs_and_accs(eps, sns, ens, hns, confs, title):
    plt.plot(confs, eps, label = 'ep', color = 'b')
    plt.plot(confs, sns, label = 'sn', color = 'r')
    plt.plot(confs, ens, label = 'en', linestyle = '-.', color = 'g')
    plt.plot(confs, hns, label = 'hn', linestyle = '-.', color = 'orange')
    plt.title(title)
    plt.xlabel(f"Confidence")
    plt.ylabel(f"acc")
    plt.legend()
    plt.show()

def collect_prec_recall_sens_npv_based_on_confidence_level_based_on_percentile(df, how = 'intarna', n_values = 15, balance = True, MIN_PERC = 1, space = 'linear', calc_ens = True):
    
    total_data = df.shape[0]
    
    if space == 'linear':
        percs_data = np.linspace(MIN_PERC, 100, n_values)[::-1]
    elif space == 'log':
        percs_data = np.logspace(np.log10(MIN_PERC), np.log10(100), n_values)[::-1]
        
    prec_nt = []
    prec_intarna = []
    prec_ens = []

    recall_nt = []
    recall_intarna = []
    recall_ens = []

    spec_nt = []
    spec_intarna = []
    spec_ens = []

    npv_nt = []
    npv_intarna = []
    npv_ens = []

    x_axis = []
    
    n_total = df.shape[0]

    for i in range(n_values):
        
        n_to_sample = int(math.ceil(percs_data[i]/100 * total_data))

        if how == 'intarna':
            subset = pd.concat([
                df.sort_values('E_norm').head(math.ceil(n_to_sample/2)), 
                df.sort_values('E_norm').tail(math.ceil(n_to_sample/2))
            ], axis = 0).reset_index(drop = True)
            
        elif how == 'nt':
            subset = pd.concat([
                df.sort_values('probability').head(math.ceil(n_to_sample/2)), 
                df.sort_values('probability').tail(math.ceil(n_to_sample/2))
            ], axis = 0).reset_index(drop = True)
            
        elif how == 'ensemble':
            subset = pd.concat([
                df.sort_values('ensemble_score').head(math.ceil(n_to_sample/2)), 
                df.sort_values('ensemble_score').tail(math.ceil(n_to_sample/2))
            ], axis = 0).reset_index(drop = True)
        else:
            raise NotImplementedError
        
        n_subset = subset.shape[0]
        
        if balance:
            subset = balance_df(subset)
        
        assert n_subset/n_total >= MIN_PERC/100
        assert abs(n_subset/n_total - percs_data[i]/100) < 0.02

        precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'probability')
        prec_nt.append(precision)
        recall_nt.append(recall)
        spec_nt.append(specificity)
        npv_nt.append(npv)

        precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'E_norm_conf')
        prec_intarna.append(precision)
        recall_intarna.append(recall)
        spec_intarna.append(specificity)
        npv_intarna.append(npv)

        if calc_ens:
            precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'ensemble_score')
            prec_ens.append(precision)
            recall_ens.append(recall)
            spec_ens.append(specificity)
            npv_ens.append(npv)
        
        value = str(np.round(percs_data[i], 2))
        x_axis.append(value)
        
    if calc_ens:
        return x_axis, (prec_nt, recall_nt, spec_nt, npv_nt), (prec_intarna, recall_intarna, spec_intarna, npv_intarna), (prec_ens, recall_ens, spec_ens, npv_ens)
    else:
        return x_axis, (prec_nt, recall_nt, spec_nt, npv_nt), (prec_intarna, recall_intarna, spec_intarna, npv_intarna)


def collect_prec_recall_sens_npv_based_on_confidence_level_based_on_treshold(df, how = 'intarna', n_values = 15, balance = True, MIN_PERC = 1, MIN_SAMPLES = 8, calc_ens = True):
    
    total_data = df.shape[0]
    
    confidence_space = np.linspace(0.51, 0.99, n_values)
        
    prec_nt = []
    prec_intarna = []
    prec_ens = []

    recall_nt = []
    recall_intarna = []
    recall_ens = []

    spec_nt = []
    spec_intarna = []
    spec_ens = []

    npv_nt = []
    npv_intarna = []
    npv_ens = []

    merged_x_axis = []
    
    n_total = df.shape[0]

    for i in range(n_values):
        
        treshold = confidence_space[i]

        if how == 'intarna':
            subset = df[(df.E_norm_conf > treshold) | (df.E_norm_conf < (1-treshold))].reset_index(drop = True)
        elif how == 'nt':
            subset = df[(df.probability > treshold) | (df.probability < (1-treshold))].reset_index(drop = True)
        elif how == 'ensemble':
            subset = df[(df.ensemble_score > treshold) | (df.ensemble_score < (1-treshold))].reset_index(drop = True)
        else:
            raise NotImplementedError
        
        n_subset = subset.shape[0]
            
        calc = False
        if set(subset.ground_truth.value_counts().index) == set([0, 1]):
            n_samples_min = min(subset.ground_truth.value_counts()[0], subset.ground_truth.value_counts()[1])
            if ( (n_subset/n_total)*100 > MIN_PERC ) & (n_samples_min >= MIN_SAMPLES):
                calc = True
        
        
        if balance:
            subset = balance_df(subset)
        
        if calc:
            precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'probability')
        else:
            precision, recall, specificity, npv = np.nan, np.nan, np.nan, np.nan
        prec_nt.append(precision)
        recall_nt.append(recall)
        spec_nt.append(specificity)
        npv_nt.append(npv)
        
        if calc:
            precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'E_norm_conf')
        else:
            precision, recall, specificity, npv = np.nan, np.nan, np.nan, np.nan
        prec_intarna.append(precision)
        recall_intarna.append(recall)
        spec_intarna.append(specificity)
        npv_intarna.append(npv)

        if calc_ens:
            if calc:
                precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'ensemble_score')
            else:
                precision, recall, specificity, npv = np.nan, np.nan, np.nan, np.nan
            prec_ens.append(precision)
            recall_ens.append(recall)
            spec_ens.append(specificity)
            npv_ens.append(npv)
            
        tuple_to_print = (np.round(confidence_space[i],2), np.round(n_subset/n_total, 2))
        merged_x_axis.append('\n\n'.join(str(x) for x in tuple_to_print))
        
    if calc_ens:
        return merged_x_axis, (prec_nt, recall_nt, spec_nt, npv_nt), (prec_intarna, recall_intarna, spec_intarna, npv_intarna), (prec_ens, recall_ens, spec_ens, npv_ens)
    else:
        return merged_x_axis, (prec_nt, recall_nt, spec_nt, npv_nt), (prec_intarna, recall_intarna, spec_intarna, npv_intarna)

def calc_prec_rec_sens_npv(subset, column):
    """
    Calculate precision, recall, specificity, and NPV for binary classification.

    Parameters:
    subset (DataFrame): The subset of data containing 'column' and 'ground_truth' columns.
    column (str): The name of the column representing the predicted values.

    Returns:
    precision (float): Precision score.
    recall (float): Recall score.
    specificity (float): Specificity score.
    npv (float): Negative predictive value score.
    """

    tp = np.sum((subset[column] > 0.5) & (subset.ground_truth == 1))
    fn = np.sum((subset[column] < 0.5) & (subset.ground_truth == 1))
    fp = np.sum((subset[column] > 0.5) & (subset.ground_truth == 0))
    tn = np.sum((subset[column] < 0.5) & (subset.ground_truth == 0))

    precision = np.round(tp / (tp + fp), 2) if (tp + fp > 0) else np.nan
    recall = np.round(tp / (tp + fn), 2) if (tp + fn > 0) else np.nan
    specificity = np.round(tn / (tn + fp), 2) if (tn + fp > 0) else np.nan
    npv = np.round(tn / (tn + fn), 2) if (tn + fn > 0) else np.nan

    return precision, recall, specificity, npv


def collect_prec_recall_sens_npv_based_on_confidence_level_based_on_treshold(df, how = 'intarna', n_values = 15, balance = True, MIN_PERC = 1, MIN_SAMPLES = 8, calc_ens = True):
    
    confidence_space = np.linspace(0.51, 0.99, n_values)
        
    prec_nt = []
    prec_intarna = []
    prec_ens = []

    recall_nt = []
    recall_intarna = []
    recall_ens = []

    spec_nt = []
    spec_intarna = []
    spec_ens = []

    npv_nt = []
    npv_intarna = []
    npv_ens = []

    merged_x_axis = []
    
    n_total = df.shape[0]

    for i in range(n_values):
        
        treshold = confidence_space[i]

        if how == 'intarna':
            subset = df[(df.E_norm_conf > treshold) | (df.E_norm_conf < (1-treshold))].reset_index(drop = True)
        elif how == 'nt':
            subset = df[(df.probability > treshold) | (df.probability < (1-treshold))].reset_index(drop = True)
        elif how == 'ensemble':
            subset = df[(df.ensemble_score > treshold) | (df.ensemble_score < (1-treshold))].reset_index(drop = True)
        else:
            raise NotImplementedError
        
        n_subset = subset.shape[0]
            
        calc = False
        if set(subset.ground_truth.value_counts().index) == set([0, 1]):
            n_samples_min = min(subset.ground_truth.value_counts()[0], subset.ground_truth.value_counts()[1])
            if ( (n_subset/n_total)*100 > MIN_PERC ) & (n_samples_min >= MIN_SAMPLES):
                calc = True
        
        
        if balance:
            subset = balance_df(subset)
        
        if calc:
            precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'probability')
        else:
            precision, recall, specificity, npv = np.nan, np.nan, np.nan, np.nan
        prec_nt.append(precision)
        recall_nt.append(recall)
        spec_nt.append(specificity)
        npv_nt.append(npv)
        
        if calc:
            precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'E_norm_conf')
        else:
            precision, recall, specificity, npv = np.nan, np.nan, np.nan, np.nan
        prec_intarna.append(precision)
        recall_intarna.append(recall)
        spec_intarna.append(specificity)
        npv_intarna.append(npv)

        if calc_ens:
            if calc:
                precision, recall, specificity, npv = calc_prec_rec_sens_npv(subset, 'ensemble_score')
            else:
                precision, recall, specificity, npv = np.nan, np.nan, np.nan, np.nan
            prec_ens.append(precision)
            recall_ens.append(recall)
            spec_ens.append(specificity)
            npv_ens.append(npv)
            
        tuple_to_print = (np.round(confidence_space[i],2), np.round(n_subset/n_total, 2))
        merged_x_axis.append('\n\n'.join(str(x) for x in tuple_to_print))
        
    if calc_ens:
        return merged_x_axis, (prec_nt, recall_nt, spec_nt, npv_nt), (prec_intarna, recall_intarna, spec_intarna, npv_intarna), (prec_ens, recall_ens, spec_ens, npv_ens)
    else:
        return merged_x_axis, (prec_nt, recall_nt, spec_nt, npv_nt), (prec_intarna, recall_intarna, spec_intarna, npv_intarna)

    
def collect_results_based_on_confidence_level_based_on_treshold(df, how = 'intarna', n_values = 15, balance = True, MIN_PERC = 1, MIN_SAMPLES = 8, calc_ens = True):
    
    
    confidence_space = np.linspace(0.51, 0.99, n_values)
                                                                
        
    auc_nt = []
    auc_intarna = []
    auc_ens = []
    merged_x_axis = []
    
    n_total = df.shape[0]
    
    for i in range(n_values):
        
        treshold = confidence_space[i]

        if how == 'intarna':
            subset = df[(df.E_norm_conf > treshold) | (df.E_norm_conf < (1-treshold))].reset_index(drop = True)
        elif how == 'nt':
            subset = df[(df.probability > treshold) | (df.probability < (1-treshold))].reset_index(drop = True)
        elif how == 'ensemble':
            subset = df[(df.ensemble_score > treshold) | (df.ensemble_score < (1-treshold))].reset_index(drop = True)
        
        n_subset = subset.shape[0]  
                                                                
        calc = False
        if set(subset.ground_truth.value_counts().index) == set([0, 1]):
            n_samples_min = min(subset.ground_truth.value_counts()[0], subset.ground_truth.value_counts()[1])
            if ( (n_subset/n_total)*100 > MIN_PERC ) & (n_samples_min >= MIN_SAMPLES):
                calc = True
                        
        
        if balance:
            subset = balance_df(subset)
                                            
        if calc:                                            
            fpr, tpr, _ = roc_curve(subset.ground_truth, subset.probability)
            roc_auc = auc(fpr, tpr)
            auc_nt.append(roc_auc) 
            
            fpr, tpr, _ = roc_curve(abs(1 - subset.ground_truth), subset.E_norm)
            roc_auc = auc(fpr, tpr)
            auc_intarna.append(roc_auc)
            
            if calc_ens:
                fpr, tpr, _ = roc_curve(subset.ground_truth, subset.ensemble_score)
                roc_auc = auc(fpr, tpr)
                auc_ens.append(roc_auc)  
        else:
            auc_nt.append(np.nan) 
            auc_intarna.append(np.nan) 
            if calc_ens:
                auc_ens.append(np.nan) 
            
        tuple_to_print = (np.round(confidence_space[i],2), np.round(subset.shape[0]/n_total * 100, 2))
            
        merged_x_axis.append('\n\n'.join(str(x) for x in tuple_to_print))
        
    if calc_ens:
        return merged_x_axis, auc_nt, auc_intarna, auc_ens
    else:
        return merged_x_axis, auc_nt, auc_intarna
