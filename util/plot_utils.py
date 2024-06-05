import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.utils import resample
from .misc import balance_df, undersample_df, is_unbalanced, obtain_majority_minority_class
from .colors import *


'''
------------------------------------------------------------------------------------------
Result plots:
'''

def plot_roc_curves_with_undersampling(models, ground_truth, n_runs=50):
    unbalanced = is_unbalanced(pd.DataFrame({'ground_truth': ground_truth}))
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')  # Plotting the random guessing line

    for model in models:
        aucs = []
        fprs = []
        tprs = []

        if unbalanced:
            for _ in range(n_runs):
                majority_class, minority_class = obtain_majority_minority_class(
                    pd.DataFrame(ground_truth).rename({0:'ground_truth'}, axis = 1)
                )

                # Undersample majority class
                majority_undersampled_idx = resample(majority_class.index, 
                                                     replace=False, 
                                                     n_samples=len(minority_class), 
                                                     random_state=np.random.randint(10000))

                # Combine minority class with undersampled majority class
                undersampled_idx = minority_class.index.union(majority_undersampled_idx)
                balanced_subset = ground_truth.loc[undersampled_idx]
                probabilities = model['prob'].loc[undersampled_idx]

                fpr, tpr, _ = roc_curve(balanced_subset, probabilities)
                fprs.append(fpr)
                tprs.append(tpr)
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
            
            # Average the FPR and TPR
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)], axis=0)
            mean_auc = np.mean(aucs)
        else:
            fpr, tpr, _ = roc_curve(ground_truth, model['prob'])
            mean_fpr = fpr
            mean_tpr = tpr
            mean_auc = auc(fpr, tpr)

        # Ensure the ROC curve starts at (0, 0) and ends at (1, 1)
        mean_fpr[0], mean_tpr[0] = 0.0, 0.0
        mean_fpr[-1], mean_tpr[-1] = 1.0, 1.0
        
        plt.plot(mean_fpr, mean_tpr, label=f'{model["model_name"]} (AUC = {mean_auc:.2f})')

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
    
    
def plot_intarna_Enorm_curves(res, treshold_plot = -100):
    sns.kdeplot(res[(res.policy == 'easypos')&(res.E_norm>treshold_plot)].E_norm, label = 'easypos')
    sns.kdeplot(res[(res.policy == 'easyneg')&(res.E_norm>treshold_plot)].E_norm, label = 'easyneg')
    sns.kdeplot(res[(res.policy == 'hardneg')&(res.E_norm>treshold_plot)].E_norm, label = 'hardneg')
    sns.kdeplot(res[(res.policy == 'smartneg')&(res.E_norm>treshold_plot)].E_norm, label = 'smartneg')
    plt.title(f'Intarna Normalized Energy distribution (easypos&smartneg)')
    plt.legend()
    plt.show()
    
def plot_ROC_based_on_confidence(df, how = 'intarna', treshold = 0.05, balance = False):
    if how == 'intarna':
        subset = df[
            (df.E_norm <= df.E_norm.quantile(treshold))|
            (df.E_norm >= df.E_norm.quantile(1-treshold))
        ]
    elif how == 'nt':
        subset = df[
            (df.probability <= treshold)|
            (df.probability >= (1-treshold))
        ]
    else:
        raise NotImplementedError
    print('perc of the total data: ', np.round(subset.shape[0]/df.shape[0], 3)*100, '%')
    if balance:
        subset = balance_df(subset)
    plot_roc_curves([{'prob': subset.probability, 'model_name': 'NT'},
                 {'prob': abs(subset.E_norm), 'model_name': 'Intarna'}
                ], subset.ground_truth)
    
    
def obtain_auc_and_perc_in_specific_treshold(treshold, res, test500, balance = True, intarna = False, ensemble = False):
    original_res_shape = res.shape[0]
    #take only big windows
    subset = test500[ (abs(test500.seed_x1 - test500.seed_x2) >treshold) & (abs(test500.seed_y1 - test500.seed_y2) > treshold) ]
    res = res[res.id_sample.isin(subset.couples)]
    
    perc = np.round((res.shape[0] / original_res_shape)*100, 2)
    
    if balance:
        res = balance_df(res)
    
    if intarna:
        fpr, tpr, _ = roc_curve(abs(1 - res.ground_truth), res.E_norm)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, _ = roc_curve(res.ground_truth, res.probability)
        roc_auc = auc(fpr, tpr)
        
    if ensemble:
        fpr, tpr, _ = roc_curve(res.ground_truth, res.ensemble_score)
        roc_auc = auc(fpr, tpr)
    
    return roc_auc, perc


def calc_metric(df, column, metric = 'precision_recall_curve'):

    if metric in ['precision', 'npv', 'recall', 'specificity', 'f1']:
        precision, recall, specificity, npv = calc_prec_rec_sens_npv(df, column)
        if metric == 'precision':
            output=precision
        elif metric == 'npv':
            output=npv
        elif metric == 'recall':
            output=recall
        elif metric == 'specificity':
            output=specificity
        elif metric == 'f1':
            output = 2* (precision*recall)/(precision+recall)

    elif metric == 'precision_recall_curve':
        try:
            precision, recall, _ = precision_recall_curve(df['ground_truth'], df[column])
            output = auc(recall, precision)
        except:
            output = np.nan

    else:
        raise NotImplementedError

    return output

def make_calculation(calc_pos, calc_neg, metric): 
    calc = False
    if (metric == 'precision'):
        if(calc_pos):
            calc = True
    elif (metric == 'npv'):
        if (calc_neg):
            calc = True
    elif metric in ['recall', 'specificity', 'f1', 'precision_recall_curve']:
        if (calc_pos)&(calc_neg):
            calc = True
    else:
        raise NotImplementedError
        
    return calc
    

def collect_results_based_on_confidence_level_based_on_treshold(df, how = 'intarna', n_values = 15, n_run_undersampling = 30, MIN_PERC = 0.05, metric = 'precision_recall_curve', calc_ens = True, consensus = False):
    
    auc_nt = []
    auc_intarna = []
    auc_ens = []
    percentages = []
    conf_space_list = []
    
    n_total = df.shape[0]
    
    confidence_space = np.linspace(0.51, 0.99, n_values)
    for i in range(n_values):
        
        treshold = confidence_space[i]

        if how == 'intarna':
            column = 'E_norm_conf'
        elif how == 'nt':
            column = 'probability'
        elif how == 'ensemble':
            column = 'ensemble_score'
        else:
            raise NotImplementedError
            
        pred_pos, pred_neg = select_predictions(df, treshold, column, consensus)
            
        calc_pos = should_I_calculate(pred_pos.shape[0], MIN_PERC, n_total)
        calc_neg = should_I_calculate(pred_neg.shape[0], MIN_PERC, n_total)
        
        calc = make_calculation(calc_pos, calc_neg, metric)
        if calc:
            auc_score_nt = []
            auc_score_intarna = []
            auc_score_ens = []

            for _ in range(n_run_undersampling):
                # Undersample the larger DataFrame to match the size of the smaller one       
                
                if how == 'intarna':                    
                    df_concatenated_nt = obtain_df_concatenated(pred_pos[pred_pos['probability'] > 0.5].reset_index(drop = True), pred_neg[pred_neg['probability'] < 0.5].reset_index(drop = True))
                    df_concatenated_intarna = obtain_df_concatenated(pred_pos, pred_neg)
                    if calc_ens:
                        df_concatenated_ens = obtain_df_concatenated(pred_pos[pred_pos['ensemble_score'] > 0.5].reset_index(drop = True), pred_neg[pred_neg['ensemble_score'] < 0.5].reset_index(drop = True))
                
                elif how == 'nt':
                    df_concatenated_nt = obtain_df_concatenated(pred_pos, pred_neg)
                    df_concatenated_intarna = obtain_df_concatenated(pred_pos[pred_pos['E_norm_conf'] > 0.5].reset_index(drop = True), pred_neg[pred_neg['E_norm_conf'] < 0.5].reset_index(drop = True))
                    if calc_ens:
                        df_concatenated_ens = obtain_df_concatenated(pred_pos[pred_pos['ensemble_score'] > 0.5].reset_index(drop = True), pred_neg[pred_neg['ensemble_score'] < 0.5].reset_index(drop = True))
                        
                elif how == 'ensemble':
                    df_concatenated_nt = obtain_df_concatenated(pred_pos[pred_pos['probability'] > 0.5].reset_index(drop = True), pred_neg[pred_neg['probability'] < 0.5].reset_index(drop = True))
                    df_concatenated_intarna = obtain_df_concatenated(pred_pos[pred_pos['E_norm_conf'] > 0.5].reset_index(drop = True), pred_neg[pred_neg['E_norm_conf'] < 0.5].reset_index(drop = True))
                    if calc_ens:
                        df_concatenated_ens = obtain_df_concatenated(pred_pos, pred_neg)
                        
                auc_score_nt.append(calc_metric(df_concatenated_nt, 'probability', metric = metric))
                auc_score_intarna.append(calc_metric(df_concatenated_intarna, 'E_norm_conf', metric = metric))
                if calc_ens:
                    auc_score_ens.append(calc_metric(df_concatenated_ens, 'ensemble_score', metric = metric))

            auc_nt.append(np.mean(auc_score_nt))
            auc_intarna.append(np.mean(auc_score_intarna))
            if calc_ens:
                auc_ens.append(np.mean(auc_score_ens))
                
            perc_nt, perc_intarna, perc_ens = len(df_concatenated_nt)/n_total * 100, len(df_concatenated_intarna)/n_total * 100, len(df_concatenated_ens)/n_total * 100
        else:
            auc_nt.append(np.nan) 
            auc_intarna.append(np.nan) 
            if calc_ens:
                auc_ens.append(np.nan) 
            
            perc_nt, perc_intarna, perc_ens = 0, 0, 0
        
        percentages.append([perc_nt, perc_intarna, perc_ens])
        conf_space_list.append(str(np.round(confidence_space[i],2)))
        
    if calc_ens:
        return conf_space_list, percentages, auc_nt, auc_intarna, auc_ens
    else:
        return conf_space_list, percentages, auc_nt, auc_intarna
    
    
    

def select_predictions(df, threshold, column, consensus):
    if consensus:
        pred_pos = df[(df['probability'] > threshold) & (df['E_norm_conf'] > threshold)].reset_index(drop=True)
        pred_neg = df[(df['probability'] < (1 - threshold)) & (df['E_norm_conf'] < (1 - threshold))].reset_index(drop=True)
    else:
        pred_pos = df[df[column] > threshold].reset_index(drop=True)
        pred_neg = df[df[column] < (1 - threshold)].reset_index(drop=True)
    return pred_pos, pred_neg


def collect_results_based_on_confidence_level_based_on_percentile(df, how = 'intarna', n_values = 15, balance = False, MIN_PERC = 1, space = 'linear', calc_ens = True):
    
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

def collect_prec_recall_sens_npv_based_on_confidence_level_based_on_percentile(df, how = 'intarna', n_values = 15, balance = False, MIN_PERC = 1, space = 'linear', calc_ens = True):
    
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

def plot_results_based_on_percentile(subset, task, MIN_PERC, space, n_values = 12, size_multiplier = 10, plot_ens = False):

    confidence_level, auc_nt, _, _ = collect_results_based_on_confidence_level_based_on_percentile(subset, how = 'nt', MIN_PERC = MIN_PERC, balance = False, n_values = n_values, space = space)
    _, _, auc_intarna, _ = collect_results_based_on_confidence_level_based_on_percentile(subset, how = 'intarna', MIN_PERC = MIN_PERC, balance = False, n_values = n_values, space = space)
    _, _, _, auc_ens = collect_results_based_on_confidence_level_based_on_percentile(subset, how = 'ensemble', MIN_PERC = MIN_PERC, balance = False, n_values = n_values, space = space)


    plt.plot(confidence_level, auc_nt, label = 'nt')
    plt.plot(confidence_level, auc_intarna, label = 'intarna')

    if plot_ens:
        plt.plot(confidence_level, auc_ens, label = 'ensemble')
    plt.title(f'AUC based on respective Confidence Levels, task: {task}')
    plt.legend()
    plt.ylabel('AUC')
    plt.xlabel(f"Perc%")
    plt.show()

    subset = balance_df(subset)


    confidence_level, (prec_nt, recall_nt, spec_nt, npv_nt), (_, _, _, _), (_, _, _, _) = collect_prec_recall_sens_npv_based_on_confidence_level_based_on_percentile(subset, how = 'nt', MIN_PERC = MIN_PERC, balance = False, n_values = n_values)
    _, (_, _, _, _), (prec_intarna, recall_intarna, spec_intarna, npv_intarna), (_, _, _, _) = collect_prec_recall_sens_npv_based_on_confidence_level_based_on_percentile(subset, how = 'intarna', MIN_PERC = MIN_PERC, balance = False, n_values = n_values)
    _, (_, _, _, _), (_, _, _, _), (prec_ens, recall_ens, spec_ens, npv_ens) = collect_prec_recall_sens_npv_based_on_confidence_level_based_on_percentile(subset, how = 'ensemble', MIN_PERC = MIN_PERC, balance = False, n_values = n_values)


    plot_prec_npv_based_on_treshold_confidence(confidence_level, prec_nt, npv_nt, prec_intarna, npv_intarna, prec_ens, npv_ens, plot_ens, task)
    

def should_I_calculate(n_subset, MIN_PERC, n_total):
    
    MIN_SAMPLES = int((n_subset * MIN_PERC) / 100)

    if (n_subset/n_total)*100 > MIN_PERC:
        calc = True
    
    else:
        calc = False
        
    return calc

def get_prec_npv(pred_pos, pred_neg, column, calc_pos, calc_neg):
    if calc_pos:
        precision, _, _, _ = calc_prec_rec_sens_npv(pred_pos, column)
    else:
        precision = np.nan
    if calc_neg:
        _, _, _, npv = calc_prec_rec_sens_npv(pred_neg, column)
    else:
        npv = np.nan
        
    return precision, npv

def collect_prec_npv_based_on_confidence_level_based_on_treshold(df, how = 'intarna', n_values = 15, MIN_PERC = 1, MIN_SAMPLES = 8, calc_ens = True):
    
    confidence_space = np.linspace(0.51, 0.99, n_values)
        
    prec_nt = []
    prec_intarna = []
    prec_ens = []

    npv_nt = []
    npv_intarna = []
    npv_ens = []

    merged_x_axis_pos = []
    merged_x_axis_neg = []
    
    n_total = df.shape[0]

    for i in range(n_values):
        
        treshold = confidence_space[i]

        if how == 'intarna':
            pred_pos = df[(df.E_norm_conf > treshold)].reset_index(drop = True)
            pred_neg = df[(df.E_norm_conf < (1-treshold))].reset_index(drop = True)
        elif how == 'nt':
            pred_pos = df[(df.probability > treshold)].reset_index(drop = True)
            pred_neg = df[(df.probability < (1-treshold))].reset_index(drop = True)
        elif how == 'ensemble':
            pred_pos = df[(df.ensemble_score > treshold)].reset_index(drop = True)
            pred_neg = df[(df.ensemble_score < (1-treshold))].reset_index(drop = True)
        else:
            raise NotImplementedError
            
        calc_pos = should_I_calculate(pred_pos.shape[0], MIN_PERC, n_total)
        calc_neg = should_I_calculate(pred_neg.shape[0], MIN_PERC, n_total)
        
        precision, npv = get_prec_npv(pred_pos, pred_neg, 'probability', calc_pos, calc_neg)
        prec_nt.append(precision)
        npv_nt.append(npv)
        
        precision, npv = get_prec_npv(pred_pos, pred_neg, 'E_norm_conf', calc_pos, calc_neg)
        prec_intarna.append(precision)
        npv_intarna.append(npv)

        if calc_ens:
            precision, npv = get_prec_npv(pred_pos, pred_neg, 'ensemble_score', calc_pos, calc_neg)
            prec_ens.append(precision)
            npv_ens.append(npv)
            
        n_subset_pos = pred_pos.shape[0]
        n_subset_neg = pred_neg.shape[0]
        conf_to_print = np.round(confidence_space[i],2)
        tuple_to_print = (conf_to_print, np.round(n_subset_pos/n_total * 100, 2))
        merged_x_axis_pos.append('\n\n'.join(str(x) for x in tuple_to_print))
        tuple_to_print = (conf_to_print, np.round(n_subset_neg/n_total * 100, 2))
        merged_x_axis_neg.append('\n\n'.join(str(x) for x in tuple_to_print))
        
    if calc_ens:
        return (merged_x_axis_pos, merged_x_axis_neg), (prec_nt, npv_nt), (prec_intarna, npv_intarna), (prec_ens, npv_ens)
    else:
        return (merged_x_axis_pos, merged_x_axis_neg), (prec_nt, npv_nt), (prec_intarna, npv_intarna)
    
    



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

    
def collect_results_based_on_confidence_level_based_on_treshold_old(df, how = 'intarna', n_values = 15, balance = True, MIN_PERC = 1, MIN_SAMPLES = 8, calc_ens = True):
    
    
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

    
def get_policies_list(task):
    """Returns policies based on task."""
    if task == 'patches':
        return ['hardneg', 'easyneg', 'easypos']
    elif task == 'interactors':
        return ['easypos', 'smartneg']
    else:
        raise NotImplementedError
        
def unzip_confidence_level_percentage(confidence_level_dirty):
    confidence_level, perc = zip(*[s.split('\n\n') for s in confidence_level_dirty])
    return confidence_level, perc

def plot_metric_based_on_treshold_confidence(confidence_level, auc_nt, perc_nt, auc_intarna, perc_intarna, auc_ens, perc_ens, task, size_multiplier, plot_ens, metric):
    # Creazione del plot
    plt.figure(figsize=(10, 6))

    # Plot AUC per modello 1 e modello 2
    plt.plot(confidence_level, auc_nt, marker='o', label='NT')
    plt.plot(confidence_level, auc_intarna, marker='o', label='INTARNA')
    if plot_ens:
        plt.plot(confidence_level, auc_ens, marker='o', label='ensemble')
    
    # Opzionalmente, variare la dimensione dei punti in base alla numerosità
    for i, size in enumerate(perc_nt):
        plt.scatter(confidence_level[i], auc_nt[i], s=float(size)*size_multiplier, color=COLOR_NT_AUC)
    for i, size in enumerate(perc_intarna):
        plt.scatter(confidence_level[i], auc_intarna[i], s=float(size)*size_multiplier, color=COLOR_INTARNA_AUC)
    if plot_ens:
        for i, size in enumerate(perc_ens):
            plt.scatter(confidence_level[i], auc_ens[i], s=float(size)*size_multiplier, color=COLOR_ENS_AUC)
            
    plt.title(f'{metric} based on respective Confidence Levels, task: {task}')
    plt.xlabel('Confidence Level %')
    plt.ylabel(f'{metric}')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    
def plot_prec_npv_based_on_treshold_confidence(confidence_level, prec_nt, npv_nt, perc_pos_nt, perc_neg_nt, prec_intarna, npv_intarna, perc_pos_intarna, perc_neg_intarna, prec_ens, npv_ens, perc_pos_ens, perc_neg_ens, plot_ens, size_multiplier, task):
    
    plt.figure(figsize=(10, 6))
    # PLOT 2 COMPARE PREC, NPV
    plt.plot(confidence_level, prec_nt, label = 'Precision NT', linestyle = '--', color = COLOR_NT_PREC)
    plt.plot(confidence_level, npv_nt, label = 'NPV  NT', linestyle = '-.', color = COLOR_NT_NPV)

    plt.plot(confidence_level, prec_intarna, label = 'Precision INTARNA', linestyle = '--', color = COLOR_INTARNA_PREC)
    plt.plot(confidence_level, npv_intarna, label = 'NPV  INTARNA', linestyle = '-.', color = COLOR_INTARNA_NPV)

    if plot_ens:
        plt.plot(confidence_level, prec_ens, label = 'Precision ENSEMBLE', linestyle = '--', color = COLOR_ENS_PREC)
        plt.plot(confidence_level, npv_ens, label = 'NPV  ENSEMBLE', linestyle = '-.', color = COLOR_ENS_NPV)

    # Opzionalmente, variare la dimensione dei punti in base alla numerosità
    for i, size in enumerate(perc_pos_nt):
        plt.scatter(confidence_level[i], prec_nt[i], s=float(size)*size_multiplier, color=COLOR_NT_PREC)
    for i, size in enumerate(perc_neg_nt):
        plt.scatter(confidence_level[i], npv_nt[i], s=float(size)*size_multiplier, color=COLOR_NT_NPV)

    for i, size in enumerate(perc_pos_intarna):
        plt.scatter(confidence_level[i], prec_intarna[i], s=float(size)*size_multiplier, color=COLOR_INTARNA_PREC)
    for i, size in enumerate(perc_neg_intarna):
        plt.scatter(confidence_level[i], npv_intarna[i], s=float(size)*size_multiplier, color=COLOR_INTARNA_NPV)
    if plot_ens:
        for i, size in enumerate(perc_pos_ens):
            plt.scatter(confidence_level[i], prec_ens[i], s=float(size)*size_multiplier, color=COLOR_ENS_PREC)
        for i, size in enumerate(perc_neg_ens):
            plt.scatter(confidence_level[i], npv_ens[i], s=float(size)*size_multiplier, color=COLOR_ENS_NPV)

    plt.title(f'Precision TP/(TP+FP), NPV TN/(TN + FN) based on respective Confidence Levels, task: {task}')
    plt.legend()

    plt.ylabel('Precision, NPV')
    plt.xlabel(f"Perc%")
    plt.grid(True, alpha=0.5)

    plt.show()


def plot_results_based_on_treshold_old(subset, task, MIN_PERC, MIN_SAMPLES, n_values = 12, size_multiplier = 10, plot_ens = False, order_by = 'normal', n_run_undersampling = 30, metric = 'precision_recall_curve', consensus = False):

    assert order_by in ['normal', 'nt', 'intarna', 'ensemble']

    if order_by == 'normal':

        confidence_level, percentages, metric_nt, _, _ = collect_results_based_on_confidence_level_based_on_treshold(subset, how = 'nt', n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, calc_ens = True, consensus=consensus)
        _, percentages, _, metric_intarna, _ = collect_results_based_on_confidence_level_based_on_treshold(subset, how = 'intarna', n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, calc_ens = True, consensus=consensus)
        _, percentages, _, _, metric_ens = collect_results_based_on_confidence_level_based_on_treshold(subset, how = 'ensemble', n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, calc_ens = True, consensus=consensus)

        perc_nt, perc_intarna, perc_ens = list(map(list, zip(*percentages)))

    else:
        confidence_level, percentages, metric_nt, metric_intarna, metric_ens = collect_results_based_on_confidence_level_based_on_treshold(subset, how = order_by, n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, calc_ens = True, consensus=consensus)
        perc_nt, perc_intarna, perc_ens = list(map(list, zip(*percentages)))
    
    plot_metric_based_on_treshold_confidence(confidence_level,
    metric_nt, perc_nt, 
    metric_intarna, perc_intarna, 
    metric_ens, perc_ens, 
    task, size_multiplier, plot_ens, metric
    )

    if order_by == 'normal':
        (conflevel_pos, conflevel_neg), (prec_nt, npv_nt), (_, _), (_, _) = collect_prec_npv_based_on_confidence_level_based_on_treshold(subset, how = 'nt', MIN_PERC = MIN_PERC, MIN_SAMPLES=MIN_SAMPLES, n_values = n_values)
        (conflevel_pos_intarna, conflevel_neg_intarna), (_,  _), (prec_intarna, npv_intarna), (_, _) = collect_prec_npv_based_on_confidence_level_based_on_treshold(subset, how = 'intarna', MIN_PERC = MIN_PERC, MIN_SAMPLES=MIN_SAMPLES, n_values = n_values)
        (conflevel_pos_ens, conflevel_neg_ens), (_, _), (_, _), (prec_ens, npv_ens) = collect_prec_npv_based_on_confidence_level_based_on_treshold(subset, how = 'ensemble', MIN_PERC = MIN_PERC, MIN_SAMPLES=MIN_SAMPLES, n_values = n_values)
    
        # Splitting the strings into two lists based on the separator '\n\n'
        confidence_level, perc_pos_nt = unzip_confidence_level_percentage(conflevel_pos)
        _, perc_neg_nt = unzip_confidence_level_percentage(conflevel_neg)
        _, perc_pos_intarna = unzip_confidence_level_percentage(conflevel_pos_intarna)
        _, perc_neg_intarna = unzip_confidence_level_percentage(conflevel_neg_intarna)
        _, perc_pos_ens = unzip_confidence_level_percentage(conflevel_pos_ens)
        _, perc_neg_ens = unzip_confidence_level_percentage(conflevel_neg_ens)

    else:
        (conflevel_pos, conflevel_neg), (prec_nt, npv_nt), (prec_intarna, npv_intarna), (prec_ens, npv_ens) = collect_prec_npv_based_on_confidence_level_based_on_treshold(subset, how = order_by, MIN_PERC = MIN_PERC, MIN_SAMPLES=MIN_SAMPLES, n_values = n_values)
        confidence_level, perc_pos_nt = unzip_confidence_level_percentage(conflevel_pos)
        _, perc_neg_nt = unzip_confidence_level_percentage(conflevel_neg)
        perc_pos_intarna = perc_pos_nt
        perc_neg_intarna = perc_neg_nt
        perc_pos_ens = perc_pos_nt
        perc_neg_ens = perc_neg_nt

    plot_prec_npv_based_on_treshold_confidence(confidence_level,
        prec_nt, npv_nt, perc_pos_nt, perc_neg_nt,
        prec_intarna, npv_intarna, perc_pos_intarna, perc_neg_intarna,
        prec_ens, npv_ens, perc_pos_ens, perc_neg_ens,
        plot_ens, size_multiplier, task)

    
def get_results_based_on_treshold(subset, MIN_PERC, n_values = 12, order_by = 'normal', n_run_undersampling = 30, metric = 'precision_recall_curve', consensus = False):

    assert order_by in ['normal', 'nt', 'intarna', 'ensemble']

    if order_by == 'normal':

        confidence_level, percentages, metric_nt, _, _ = collect_results_based_on_confidence_level_based_on_treshold(subset, how = 'nt', n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, calc_ens = True, consensus=consensus)
        _, percentages, _, metric_intarna, _ = collect_results_based_on_confidence_level_based_on_treshold(subset, how = 'intarna', n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, calc_ens = True, consensus=consensus)
        _, percentages, _, _, metric_ens = collect_results_based_on_confidence_level_based_on_treshold(subset, how = 'ensemble', n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, calc_ens = True, consensus=consensus)

        perc_nt, perc_intarna, perc_ens = list(map(list, zip(*percentages)))

    else:
        confidence_level, percentages, metric_nt, metric_intarna, metric_ens = collect_results_based_on_confidence_level_based_on_treshold(subset, how = order_by, n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, calc_ens = True, consensus=consensus)
        perc_nt, perc_intarna, perc_ens = list(map(list, zip(*percentages)))
    
    return confidence_level, metric_nt, perc_nt, metric_intarna, perc_intarna, metric_ens, perc_ens

def plot_results_based_on_treshold(subset, task, MIN_PERC, MIN_SAMPLES, n_values = 12, size_multiplier = 10, plot_ens = False, order_by = 'normal', n_run_undersampling = 30, metric = 'precision_recall_curve', consensus = False):

    confidence_level, metric_nt, perc_nt, metric_intarna, perc_intarna, metric_ens, perc_ens = get_results_based_on_treshold(subset, MIN_PERC, n_values, order_by, n_run_undersampling, metric, consensus)
    
    plot_metric_based_on_treshold_confidence(confidence_level,
    metric_nt, perc_nt, 
    metric_intarna, perc_intarna, 
    metric_ens, perc_ens, 
    task, size_multiplier, plot_ens, metric
    )

    
def get_results_based_on_treshold_for_all_models(subset, MIN_PERC, how = 'nt', n_values = 12, n_run_undersampling = 30, metric = 'precision_recall_curve', balance_predictions_at_each_step = True):

    confidence_level, percentages, auc_model = collect_results_based_on_confidence_level_based_on_treshold_for_single_model(subset, how = how, n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric, balance_predictions_at_each_step = balance_predictions_at_each_step)
    
    #perc_nt, perc_intarna, perc_ens = list(map(list, zip(*percentages)))
    
    return confidence_level, auc_model, percentages

def plot_results_based_on_treshold_for_all_models(df, MIN_PERC, list_of_models_to_test, n_values = 12, n_run_undersampling = 15, metric = 'precision_recall_curve', task_name = 'patches', size_multiplier = 10, balance_predictions_at_each_step = True):

    auc_runs = []
    perc_runs = []
    
    for run in range(n_run_undersampling):
        
        subset = undersample_df(df) #undersampling at each run
        
        auc_models = []
        perc_models = []
        model_names = []

        confidence_level = []

        for model_name in list_of_models_to_test:
            c_l, auc_model, percentages = get_results_based_on_treshold_for_all_models(
                subset, MIN_PERC, how = model_name, n_values = n_values, n_run_undersampling = n_run_undersampling, metric = metric, balance_predictions_at_each_step = balance_predictions_at_each_step
            )
            auc_models.append(auc_model)
            perc_models.append(percentages)
            model_names.append(model_name)

            if len(c_l)>len(confidence_level):
                confidence_level = c_l
                
        
        auc_runs.append(auc_models)
        perc_runs.append(perc_models)
    
    auc_models=np.mean(auc_runs, axis = 0)
    perc_models=np.mean(perc_runs, axis = 0)
    
    plot_metric_confidence_for_all_models(confidence_level, auc_models, perc_models, model_names, task_name, size_multiplier, metric, string_label = 'Confidence Level')
    
def plot_metric_confidence_for_all_models(confidence_level, auc_models, perc_models, model_names, task_name, size_multiplier, metric, string_label):
    
    """
    auc_models: list
    perc_models: list
    model_names : list
    """
    
    for i, model_name in enumerate(model_names):
        model_color = model_colors_dict[model_name]
        
        plt.plot(confidence_level, auc_models[i], marker='o', label=model_name, color = model_color)
        #print(model_name, perc_models[i])
        # Opzionalmente, variare la dimensione dei punti in base alla numerosità
        for _, size in enumerate(perc_models[i]):
            plt.scatter(confidence_level[_], auc_models[i][_], s=float(size)*size_multiplier, color=model_color)
            
    plt.title(f'{metric} based on respective {string_label}, task: {task_name}')
    plt.xlabel(f'{string_label} %')
    plt.ylabel(f'{metric}')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    
    
    
def obtain_df_concatenated(pred_pos, pred_neg):
    if len(pred_pos) > len(pred_neg):
        df_undersampled = pred_pos.sample(n=len(pred_neg)).reset_index(drop = True)
        df_concatenated = pd.concat([df_undersampled, pred_neg]).reset_index(drop = True)
    else:
        df_undersampled = pred_neg.sample(n=len(pred_pos)).reset_index(drop = True)
        df_concatenated = pd.concat([pred_pos, df_undersampled]).reset_index(drop = True)
    return df_concatenated

def collect_results_based_on_confidence_level_based_on_treshold_for_single_model(df, how = 'intarna', n_values = 15, n_run_undersampling = 30, MIN_PERC = 0.05, metric = 'precision_recall_curve', balance_predictions_at_each_step = True):
    
    auc_model = []
    percentages = []
    conf_space_list = []
    
    
    confidence_space = np.linspace(0.5, 0.99, n_values)
    for i in range(n_values):
        
        treshold = confidence_space[i]

        if how == 'intarna':
            column = 'E_norm_conf'
        elif how == 'nt':
            column = 'probability'
        elif how == 'ensemble':
            column = 'ensemble_score'
        else:
            column = how
            
        pred_pos, pred_neg = select_predictions(df, treshold, column, False)
        #print(pred_pos.shape[0], pred_neg.shape[0])
            
        calc_pos = should_I_calculate(pred_pos.shape[0], MIN_PERC, df.shape[0])
        calc_neg = should_I_calculate(pred_neg.shape[0], MIN_PERC, df.shape[0])
        
        calc = make_calculation(calc_pos, calc_neg, metric)
        if calc:
            if balance_predictions_at_each_step:
                auc_score_run = []
                for _ in range(n_run_undersampling):
                    
                    # Undersample the larger DataFrame to match the size of the smaller one
                    df_concatenated = obtain_df_concatenated(pred_pos, pred_neg)
                    auc_score_run.append(calc_metric(df_concatenated, column, metric = metric))

                auc_model.append(np.mean(auc_score_run))
            else:
                df_concatenated = pd.concat([pred_pos, pred_neg]).reset_index()
                auc_model.append(calc_metric(df_concatenated, column, metric = metric))
            
            perc_model = calc_perc_model(metric, 
                                         df.shape[0],  
                                         df[df.ground_truth == 1].shape[0], 
                                         df[df.ground_truth == 0].shape[0], 
                                         pred_pos.shape[0], 
                                         pred_neg.shape[0]
                                        )
        else:
            auc_model.append(np.nan)
            perc_model = 0
        
        percentages.append(perc_model)
        conf_space_list.append(str(np.round(confidence_space[i],2)))
        
    return conf_space_list, percentages, auc_model
    
    
def calc_perc_model(metric, n_total, n_total_pos, n_total_neg, n_pred_pos, n_pred_neg):
    if metric in ['precision', 'npv',  'f1', 'precision_recall_curve']:
        perc_model = ( (n_pred_pos + n_pred_neg) / n_total ) * 100

    elif metric == 'recall':
        perc_model = ( n_pred_pos / n_total_pos ) * 100 #it can be more than 100% if n_pred_pos is > n_total_pos
        
    elif metric == 'specificity':
        perc_model = ( n_pred_neg / n_total_neg ) * 100  #it can be more than 100% if n_pred_neg is > n_total_neg
        
    return perc_model
    
    
    
def plot_length_embeddings_and_rnas(res):
    ep_len = list(res[res.policy == 'easypos'].len_emb1) + list(res[res.policy == 'easypos'].len_emb2)
    sn_len = list(res[res.policy == 'smartneg'].len_emb1) + list(res[res.policy == 'smartneg'].len_emb2)
    hn_len = list(res[res.policy == 'hardneg'].len_emb1) + list(res[res.policy == 'hardneg'].len_emb2)
    en_len = list(res[res.policy == 'easyneg'].len_emb1) + list(res[res.policy == 'easyneg'].len_emb2)

    sns.kdeplot(ep_len, label = 'ep', color = 'b')
    sns.kdeplot(sn_len, label = 'sn', color = 'r')
    sns.kdeplot(hn_len, label = 'hn', color = 'g')
    sns.kdeplot(en_len, label = 'en', color = 'orange')
    plt.title(f'Embedding Length distribution for each class')
    plt.legend()
    plt.show()


    ep_len = list(res[res.policy == 'easypos'].len_g1) + list(res[res.policy == 'easypos'].len_g2)
    sn_len = list(res[res.policy == 'smartneg'].len_g1) + list(res[res.policy == 'smartneg'].len_g2)
    hn_len = list(res[res.policy == 'hardneg'].len_g1) + list(res[res.policy == 'hardneg'].len_g2)
    en_len = list(res[res.policy == 'easyneg'].len_g1) + list(res[res.policy == 'easyneg'].len_g2)

    sns.kdeplot(ep_len, label = 'ep', color = 'b')
    sns.kdeplot(sn_len, label = 'sn', color = 'r')
    sns.kdeplot(hn_len, label = 'hn', color = 'g')
    sns.kdeplot(en_len, label = 'en', color = 'orange')
    plt.title(f'Length distribution for each class')
    plt.legend()
    plt.show()


    ep_len = list(res[res.policy == 'easypos'].original_length1) + list(res[res.policy == 'easypos'].original_length2)
    sn_len = list(res[res.policy == 'smartneg'].original_length1) + list(res[res.policy == 'smartneg'].original_length2)
    hn_len = list(res[res.policy == 'hardneg'].original_length1) + list(res[res.policy == 'hardneg'].original_length2)
    en_len = list(res[res.policy == 'easyneg'].original_length1) + list(res[res.policy == 'easyneg'].original_length2)

    sns.kdeplot(ep_len, label = 'ep', color = 'b')
    sns.kdeplot(sn_len, label = 'sn', color = 'r')
    sns.kdeplot(hn_len, label = 'hn', color = 'g')
    sns.kdeplot(en_len, label = 'en', color = 'orange')
    plt.title(f'Original Length distribution for each class')
    plt.legend()
    plt.show()
    
    
def plot_tnr_based_on_distance(test500, ephnen, bins_distance):
    
    tnrs_nt = []
    tnrs_intarna = []
    distances_axis = []
    percs = []

    for (dist1, dist2) in bins_distance:
        subset = test500[(test500.distance_from_site >= dist1) & (test500.distance_from_site <= dist2)].reset_index(drop = True)
        subset = subset[(subset.policy.isin(['hardneg', 'easyneg']))].reset_index(drop = True)

        perc_of_total_df = np.round(
            (
            subset.shape[0] / test500[test500.policy.isin(['hardneg', 'easyneg'])].shape[0] 
            ) * 100, 2)
        percs.append(perc_of_total_df)

        couples_to_keep = set(subset.couples)
        subset = ephnen[ephnen.id_sample.isin(couples_to_keep)].reset_index(drop = True)

        #NT
        tnr = (subset.ground_truth == (subset.probability > 0.5).astype(int)).sum() / subset.shape[0]
        tnrs_nt.append(tnr)

        #INTARNA
        tnr = (subset.ground_truth == (subset.E_norm_conf > 0.5).astype(int)).sum() / subset.shape[0]
        tnrs_intarna.append(tnr)

        distances_axis.append(str(np.mean([dist1, dist2]).astype(int)))
   

    plt.figure(figsize=(10, 6))
    size_multiplier = 20
    plt.title('TNR of models in the task pathces based on distance interval')
    plt.plot(distances_axis, tnrs_nt, label = 'nt', color = COLOR_NT_AUC, linewidth=2)
    plt.plot(distances_axis, tnrs_intarna, label = 'intarna', color = COLOR_INTARNA_AUC, linewidth=2)

    for i, size in enumerate(percs):
        plt.scatter(distances_axis[i], tnrs_nt[i], s=float(size)*size_multiplier, color=COLOR_NT_AUC)

    for i, size in enumerate(percs):
        plt.scatter(distances_axis[i], tnrs_intarna[i], s=float(size)*size_multiplier, color=COLOR_INTARNA_AUC)


    plt.xlabel(f"Distance Interval")
    plt.ylabel(f"TNR Patches task")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    
def quantile_bins(series, num_bins):
    # Compute quantiles
    quantiles = pd.qcut(series, q=num_bins, duplicates='drop')
    
    # Extract quantile edges
    quantile_edges = quantiles.unique().categories.values
    
    # Generate list of lists
    bins = [[q.left, q.right] for q in quantile_edges]
    
    return bins

def obtain_regression_line(data):
    x=np.arange(len(data))
    coefficients = np.polyfit(x, data, 1)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(x)
    return y_fit

def plot_confidence_based_on_distance(test500, ephnen, bins_distance):
    
    mean_confidence_nt = [] 
    mean_confidence_intarna = [] 
    mean_distances = []

    for (dist1, dist2) in bins_distance:
        subset = test500[(test500.distance_from_site >= dist1) & (test500.distance_from_site <= dist2)].reset_index(drop = True)
        subset = subset[(subset.policy.isin(['hardneg', 'easyneg']))].reset_index(drop = True)

        couples_to_keep = set(subset.couples)
        subset = ephnen[ephnen.id_sample.isin(couples_to_keep)].reset_index(drop = True)

        mean_confidence_nt.append(abs(0.5 - subset.probability.mean()))
        mean_confidence_intarna.append(abs(0.5 - subset.E_norm_conf.mean()))
        mean_distances.append(str(np.mean([dist1, dist2]).astype(int)))

    regression_nt = obtain_regression_line(mean_confidence_nt)
    regression_intarna = obtain_regression_line(mean_confidence_intarna)

    plt.figure(figsize=(10, 6))
    plt.title('Confidence of the models VS distances from the interaction site')
    plt.plot(mean_distances, mean_confidence_nt, label = 'nt', color = COLOR_NT_AUC, linewidth=2)
    plt.plot(mean_distances, regression_nt, label = 'nt', linestyle = '--', color = COLOR_NT_AUC, linewidth=2, alpha = 0.5)
    plt.plot(mean_distances, mean_confidence_intarna, label = 'intarna', color = COLOR_INTARNA_AUC, linewidth=2)
    plt.plot(mean_distances, regression_intarna, label = 'nt', linestyle = '--', color = COLOR_INTARNA_AUC, linewidth=2, alpha = 0.5)

    plt.xlabel(f"Mean Distance of the interval")
    plt.ylabel(f"Mean Confidence in the interval")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    
    
def plot_results_based_on_topbottom_for_all_models(df, MIN_PERC, list_of_models_to_test, n_values = 12, n_run_undersampling = 15, metric = 'precision', task_name = 'patches', size_multiplier = 10):
    
    
    auc_runs = []
    perc_runs = []
    
    for run in range(n_run_undersampling):
        
        subset = undersample_df(df) #undersampling at each run
        
        auc_models = []
        perc_models = []
        model_names = []

        confidence_level = []

        for model_name in list_of_models_to_test:
            auc_model, percentages = get_results_based_on_topbottom_for_all_models(
            subset, MIN_PERC, how = model_name, n_values = n_values, n_run_undersampling = n_run_undersampling, metric = metric
        )
            auc_models.append(auc_model)
            perc_models.append(percentages)
            model_names.append(model_name)

        assert perc_models[0] == perc_models[1]
        
        auc_runs.append(auc_models)
        perc_runs.append(perc_models)
    
    auc_models=np.mean(auc_runs, axis = 0)
    perc_models=np.mean(perc_runs, axis = 0)

    plot_metric_confidence_for_all_models([str(perc) for perc in perc_models[0]], auc_models, perc_models, model_names, task_name, size_multiplier, metric, string_label = 'Percentage Data')

    
    
def get_results_based_on_topbottom_for_all_models(subset, MIN_PERC, how = 'nt', n_values = 12, n_run_undersampling = 30, metric = 'precision'):

    percentages, metric_model = collect_results_based_on_confidence_level_based_on_topbottom_for_single_model(subset, how = how, n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric)
    
    #perc_nt, perc_intarna, perc_ens = list(map(list, zip(*percentages)))
    
    return metric_model, percentages
    
def collect_results_based_on_confidence_level_based_on_topbottom_for_single_model(df, how = 'intarna', n_values = 15, n_run_undersampling = 30, MIN_PERC = 0.05, metric = 'precision'):
    
    metric_model = []
    percentages = []
    
    n_total = df.shape[0]
    
    percs_data = np.linspace(MIN_PERC, 100, n_values)[::-1]

    if how == 'intarna':
        column = 'E_norm_conf'
    elif how == 'nt':
        column = 'probability'
    elif how == 'ensemble':
        column = 'ensemble_score'
    else:
        column = how

        
    pred_pos = df[df[column]>0.5].reset_index(drop = True)
    pred_neg = df[df[column]<0.5].reset_index(drop = True)
    
    for i in range(n_values):

        n_to_sample = int(math.ceil(percs_data[i]/100 * pred_pos.shape[0]))
        
        if metric == 'precision':
            subset = pred_pos.sort_values(column, ascending = False).head(n_to_sample)
            
        elif metric == 'npv':
            subset = pred_neg.sort_values(column, ascending = False).tail(n_to_sample)
            
        elif metric in ['f1', 'recall', 'specificity', 'precision_recall_curve']:
            subset = pd.concat([
                pred_pos.sort_values(column, ascending = False).head(n_to_sample),
                pred_neg.sort_values(column, ascending = False).tail(n_to_sample)
            ], axis = 0).reset_index(drop = True)
            
        
        metric_model.append(calc_metric(subset, column, metric = metric))
        
        percentages.append(np.round(percs_data[i], 2))
        
    return percentages, metric_model


def plot_results_how_many_repeats_in_pred_pos_for_all_models(subset, MIN_PERC, list_of_models_to_test, n_values = 12, both_sr = False, feature_to_search = 'Simple_repeat'):

    perc_models = []
    model_names = []
    
    confidence_level = []
    
    for model_name in list_of_models_to_test:
        c_l, percentages = how_many_repeats_in_pred_pos_for_single_model(
            subset, how = model_name, n_values = n_values, MIN_PERC = MIN_PERC, both_sr = both_sr, feature_to_search = feature_to_search
        )
        
        perc_models.append(percentages)
        model_names.append(model_name)
        
        if len(c_l)>len(confidence_level):
            confidence_level = c_l
    
    plot_metric_confidence_for_all_models(confidence_level, perc_models, [[0 for i in range(len(perc_models[0]))] for j in range(len(perc_models))], model_names, feature_to_search, 0, 'Percentage of positive predictions', string_label = 'Confidence Level')

def how_many_repeats_in_pred_pos_for_single_model(df, how = 'intarna', n_values = 15, MIN_PERC = 0.05, both_sr = False, feature_to_search = 'Simple_repeat'):

    confidence_space = np.linspace(0.5, 0.99, n_values)
    
    conf_space_list = []
    percentages = []

    for i in range(n_values):

        treshold = confidence_space[i]

        if how == 'intarna':
            column = 'E_norm_conf'
        elif how == 'nt':
            column = 'probability'
        elif how == 'ensemble':
            column = 'ensemble_score'
        else:
            column = how

        pred_pos = df[df[column] > treshold].reset_index(drop=True)
        
        if feature_to_search == 'Simple_repeat':
            label_x = 'simple_repeat1'
            label_y = 'simple_repeat2'
        else:
            raise NotImplementedError
        
        if (pred_pos.shape[0]/df.shape[0] *100 ) > MIN_PERC:
            if both_sr:
                perc_sr = pred_pos[pred_pos[label_x] & pred_pos[label_y]].shape[0] / pred_pos.shape[0] * 100
            else:
                perc_sr = pred_pos[pred_pos[label_x] | pred_pos[label_y]].shape[0] / pred_pos.shape[0] * 100
        else:
            
            perc_sr = np.nan
            
        percentages.append(perc_sr)
        conf_space_list.append(str(np.round(confidence_space[i],2)))
            
    return conf_space_list, percentages

def plot_tnr_based_on_distance_for_all_models(enhn, bins_distance, list_of_models_to_test, figsize, size_multiplier):
    
    models_tnr = []
    distances_axis = []
    percs = []

    for (dist1, dist2) in bins_distance:
        
        subset = enhn[(enhn.distance_from_site >= dist1) & (enhn.distance_from_site <= dist2)].reset_index(drop = True)

        perc_of_total_df = np.round(
            (
            subset.shape[0] / enhn.shape[0] 
            ) * 100, 2)
        percs.append(perc_of_total_df)
        distances_axis.append(str(np.mean([dist1, dist2]).astype(int)))

        models_for_this_bin = []
        for model in list_of_models_to_test:
            column = model if model!='nt' else 'probability'
            tnr = (subset.ground_truth == (subset[column] > 0.5).astype(int)).sum() / subset.shape[0]
            models_for_this_bin.append(tnr)
        models_tnr.append(models_for_this_bin)
   

    models_tnr = list(map(list, zip(*models_tnr)))
    
    plt.figure(figsize=figsize)
    plt.title('TNR of models in the task pathces based on distance interval')
    for i, model_name in enumerate(list_of_models_to_test):
        model_color = model_colors_dict[model_name]
        plt.plot(distances_axis, models_tnr[i], label = model_name, color = model_color, linewidth=2)

        for j, size in enumerate(percs):
            plt.scatter(distances_axis[j], models_tnr[i][j], s=float(size)*size_multiplier, color=model_color)

    plt.xlabel(f"Distance Interval")
    plt.ylabel(f"TNR Patches task")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    
def plot_tnr_for_all_models(list_of_models_to_test, subset, figsize, title_suffix = '', bar_width = 0.5):
    tnrs = []
    for model in list_of_models_to_test:
        column = model if model!='nt' else 'probability'
        tnr = (subset.ground_truth == (subset[column] > 0.5).astype(int)).sum() / subset.shape[0]
        tnrs.append(tnr)

    plt.figure(figsize=figsize)
    for i, model in enumerate(list_of_models_to_test):
        model_color = model_colors_dict[model]
        plt.bar(i, tnrs[i], width=bar_width, label=model, color=model_color)

    plt.xlabel('Models')
    plt.ylabel('TRN Values')
    plt.title(f'TRN Values for Different Models, {title_suffix}')
    plt.xticks(range(len(list_of_models_to_test)), list_of_models_to_test)
    plt.show()
    
def plot_confidence_based_on_distance_for_all_models(enhn, bins_distance, list_of_models_to_test, figsize):
    
    models_conifdence = []
    distances_axis = []
    regression_lines = []

    for (dist1, dist2) in bins_distance:
        
        subset = enhn[(enhn.distance_from_site >= dist1) & (enhn.distance_from_site <= dist2)].reset_index(drop = True)

        distances_axis.append(str(np.mean([dist1, dist2]).astype(int)))

        models_for_this_bin = []
        for model in list_of_models_to_test:
            column = model if model!='nt' else 'probability'
            model_conf = 0.5 - subset[column].mean()
            models_for_this_bin.append(model_conf)
        models_conifdence.append(models_for_this_bin)
   

    models_conifdence = list(map(list, zip(*models_conifdence)))
    
    
    plt.figure(figsize=figsize)
    plt.title('TNR of models in the task pathces based on distance interval')
    for i, model_name in enumerate(list_of_models_to_test):
        model_color = model_colors_dict[model_name]
    
        plt.plot(distances_axis, models_conifdence[i], label = model_name, color = model_color, linewidth=2)
        
        regression_line = obtain_regression_line(models_conifdence[i])
        plt.plot(distances_axis, regression_line, label = model_name, color = model_color, linewidth=2, linestyle = '--')

    plt.xlabel(f"Mean Distance Interval")
    plt.ylabel(f"Confidence in the Mean Distance Interval")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    
    
def plot_features_vs_risearch2_confidence(res, based_on_percentile = True, n_values = 12, figsize = (10, 6)):
    
    if based_on_percentile:
        thresholds_ricseq = np.percentile(res.risearch2, np.linspace(0, 100, n_values + 1))[:-1]
        
    else:
        thresholds_ricseq = np.linspace(0, 1, n_values)

    diz_res = {}
    for threshold_ricseq_hq in thresholds_ricseq:
        hq_pos = res[res.risearch2>threshold_ricseq_hq]
        
        features = ['perc both no protein_coding', 'perc both protein_coding']
        
        mean_values = [
            ((hq_pos['gene1_pc'] == False) & (hq_pos['gene2_pc'] == False)).sum() / hq_pos.shape[0] * 100,
            ((hq_pos['gene1_pc'] == True) & (hq_pos['gene2_pc'] == True)).sum() / hq_pos.shape[0] * 100
        ]
        
        
        
        other_features_to_plot = ['original_area', 'n_reads', 'simple_repeats', 'sine_alu', 'low_complex']

        for i in other_features_to_plot:
            features.append(i)
            mean_values.append(hq_pos[i].mean())

        diz_res[str((threshold_ricseq_hq * 100).astype(int))] = mean_values

    df_plot = pd.DataFrame.from_dict(diz_res, 'index')
    df_plot.columns = features
    df_plot = df_plot.reset_index().rename({'index':'confidence'}, axis = 1)

    for feature in features:
        plt.figure(figsize=figsize)
        plt.title(f'{feature} based on risearch2 confidence')

        model_color = model_colors_dict['risearch2']
        plt.plot(df_plot['confidence'], df_plot[feature], label = feature, color = model_color, linewidth=2)

        plt.xlabel(f"Risearch confidence")
        plt.ylabel(f"{feature}")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()