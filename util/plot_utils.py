import pandas as pd
import numpy as np
import math
import sys
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.utils import resample
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from .misc import balance_df, undersample_df, is_unbalanced, obtain_majority_minority_class, find_extension_from_savepath
from .colors import *
from .model_names_map import map_model_names, map_experiment_names

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME


def add_task_name_to_savepath(savepath, task_name):
    # Split the file path into name and extension
    base, ext = os.path.splitext(savepath)
    # Return the modified path with task_name added before the extension
    return f"{base}_{task_name}{ext}"

'''
------------------------------------------------------------------------------------------
Result plots:
'''

def plot_interacting_region_kde_paris(df, figsize=(12, 8), savepath=''):
    """
    Plots a KDE of the combined 'w' and 'h' columns in the dataframe,
    including only values above the 99th percentile of the distribution.

    Parameters:
    - df: DataFrame containing the data with columns 'w' and 'h'.
    - figsize: Tuple specifying the size of the figure.
    - savepath: Optional path to save the plot. If empty, the plot won't be saved.
    """
    # Concatenate the columns 'w' and 'h', reset the index
    combined_data = pd.concat([df['w'], df['h']], axis=0).reset_index(drop=True)
    
    # Calculate the 99th percentile
    threshold = combined_data.quantile(0.99)
    
    # Filter the data to include only values above the 99th percentile
    filtered_data = combined_data[combined_data <= threshold]
    
    # Plotting the KDE
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(filtered_data, color="skyblue", ax=ax, linewidth=2)
    
    # Adding labels and title
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("KDE Plot of Values Above the 99th Percentile")
    
    if savepath:
        extension = savepath.split('.')[-1]
        plt.savefig(savepath, format=extension, dpi = 300, bbox_inches='tight')
    
    plt.show()


def plot_interacting_region_hist_paris(df, figsize=(12, 8), savepath=''):
    """
    Plots a histogram of the combined 'w' and 'h' columns in the dataframe, 
    including only values above the 99th percentile of the distribution.

    Parameters:
    - df: DataFrame containing the data with columns 'w' and 'h'.
    - figsize: Tuple specifying the size of the figure.
    - savepath: Optional path to save the plot. If empty, the plot won't be saved.
    """
    # Concatenate the columns 'w' and 'h', reset the index
    combined_data = pd.concat([df['w'], df['h']], axis=0).reset_index(drop=True)
    
    # Calculate the 99th percentile
    threshold = combined_data.quantile(0.99)
    
    # Filter the data to include only values above the 99th percentile
    filtered_data = combined_data[combined_data <= threshold]
    
    # Plotting the histogram
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(filtered_data, bins=40, color='skyblue', edgecolor='black', rwidth=0.8)
    
    # Adding labels and title
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Values Above the 99th Percentile")
    
    if savepath:
        extension = find_extension_from_savepath(savepath)
        plt.savefig(savepath, format=f"{extension}", dpi = 300, bbox_inches='tight')
    
    plt.show()

def draw_arrow(ax, start, end, color='black'):
    """Disegna una freccia tra due punti."""
    ax.annotate('', 
                xy=end, xytext=start, 
                arrowprops=dict(arrowstyle='<|-|>', color=color, lw=1.5))

def draw_triangle_with_arrows(side_length = 0.5, figsize=(7, 7), savepath = ''):
    # Calcolo dei vertici del triangolo equilatero con lato 3
    height = np.sqrt(3) / 2 * side_length  # Altezza del triangolo equilatero

    vertices = np.array([
        [0, 0],                              # Vertice A
        [side_length, 0],                    # Vertice B
        [side_length / 2, height]            # Vertice C
    ])

    
    # Creazione della figura
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    
    # Disegna le frecce (archi bidirezionali)
    draw_arrow(ax, vertices[0], vertices[1])  # A -> B
    draw_arrow(ax, vertices[1], vertices[0])  # B -> A
    
    draw_arrow(ax, vertices[1], vertices[2])  # B -> C
    draw_arrow(ax, vertices[2], vertices[1])  # C -> B
    
    draw_arrow(ax, vertices[2], vertices[0])  # C -> A
    draw_arrow(ax, vertices[0], vertices[2])  # A -> C

    # Aggiungi i vertici e le etichette con posizioni regolate
    labels = ['Data Quality', 'Model Performance', 'Model Confidence']
    labels = ['', '', '']
    for i, (x, y) in enumerate(vertices):
        if i == 0:  # Sposta "Data Quality" in basso e leggermente a sinistra
            ax.text(x - 0.1, y - 0.1, labels[i], ha='center', fontsize=7)
        elif i == 1:  # Sposta "Model Performance" in basso e leggermente a destra
            ax.text(x + 0.1, y - 0.1, labels[i], ha='center', fontsize=7)
        else:  # Mantieni "Model Confidence" in alto
            ax.text(x, y + 0.05, labels[i], ha='center', fontsize=7)

    
    # Aggiungi le scritte tra i vertici con posizioni regolate
    edge_labels = {
        (1, 2): (r"$\rho_{PC}$", 0.1, 0),  # Spostato leggermente a destra (+0.1 su x)
        (0, 1): (r"$\rho_{QP}$", 0, -0.1), # Spostato leggermente in basso (-0.1 su y)
        (0, 2): (r"$\rho_{QC}$", -0.1, 0), # Spostato leggermente a sinistra (-0.1 su x)
    }

    for (start, end), (label, x_offset, y_offset) in edge_labels.items():
        mid_x = (vertices[start][0] + vertices[end][0]) / 2 + x_offset  # Offset su x
        mid_y = (vertices[start][1] + vertices[end][1]) / 2 + y_offset  # Offset su y
        ax.text(mid_x, mid_y, label, fontsize=7, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Personalizzazione degli assi
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')  # Nascondi gli assi
    
    if savepath:
        extension = find_extension_from_savepath(savepath)
        plt.savefig(savepath, format=f"{extension}", dpi = 300, bbox_inches='tight')
    
    plt.show()    

def plot_bar_n_reads_hist(df, upper = 10, figsize=(12, 8), savepath=''):
    """
    Plots a histogram of the 'n_reads' column in the dataframe for interacting reads, 
    with a cutoff at 10 to show "≥ 10" on the x-axis.

    Parameters:
    - df: DataFrame containing the data. It should have a column 'n_reads' and a boolean column 'interacting'.
    - figsize: Tuple specifying the size of the figure.
    - savepath: Optional path to save the plot. If empty, the plot won't be saved.
    """
    
    # Filter for interacting reads and clip values to a maximum of 10
    n_reads_distribution = df[df['interacting']]['n_reads'].clip(upper=upper)
    
    # Determine the minimum number of reads in the data
    min_reads = int(n_reads_distribution.min())
    
    # Define bins from min_reads to upper
    bins = list(range(min_reads, upper+2))
    
    # Plotting the histogram
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(n_reads_distribution, bins=bins, align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    
    # Customizing the x-axis labels, including "≥ 10" for the last bin
    xticks_labels = list(range(min_reads, upper)) + [f'≥ {upper}'] + ['']
    ax.set_xticks(bins)
    ax.set_xticklabels(xticks_labels)
    
    # Adding labels and title
    ax.set_xlabel("Number of Reads")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Number of Reads")
    
    # Saving the plot if a savepath is provided
    if savepath:
        extension = find_extension_from_savepath(savepath)
        plt.savefig(savepath, format=f"{extension}", dpi = 300, bbox_inches='tight')
    
    plt.show()

def plot_matrix_area_kde_curves(datasets, labels, fontsize = 7, figsize=(10, 6), legend_fontsize = 5, xlabel="Square Root of Contact Matrix Area", title="KDE Plot of Contact Matrix Area", savepath=''):
    """
    Plots KDE curves for the square root of the product of length_1 and length_2 in multiple datasets,
    after filtering out values greater than the 99th percentile for each distribution.
    
    Parameters:
    - datasets: List of pandas DataFrames containing 'length_1' and 'length_2' columns
    - labels: List of labels corresponding to each dataset for the legend
    - xlabel: Label for the x-axis
    - title: Title of the plot
    - savepath: Path to save the plot, with the extension indicating the format (e.g., 'plot.png')
    """
    
    plt.figure(figsize=figsize)
    
    for data, label in zip(datasets, labels):
        # Calculate the square root of product of lengths and filter by the 99th percentile
        filtered_data = np.sqrt(data.length_1 * data.length_2)
        filtered_data = filtered_data[filtered_data <= np.percentile(filtered_data, 99)]
        
        # Plot KDE curve
        sns.kdeplot(filtered_data, label=label, shade=False)
    
    # Add legend and labels
    plt.legend(fontsize=legend_fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel("Density", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    
    # Save plot if savepath is provided
    if savepath:
        extension = find_extension_from_savepath(savepath)
        plt.savefig(savepath, format=f"{extension}", dpi = 300, bbox_inches='tight')
    
    # Show plot
    plt.show()

def plot_interaction_region(df, title = 'Full set', savepath = '', plot_legend = False, figsize=(12, 8)):
    
    df['where'] = df['where'].apply(lambda x: x.replace("UTR5", "5'UTR"))
    df['where'] = df['where'].apply(lambda x: x.replace("UTR3", "3'UTR"))
    df['where'] = df['where'].apply(lambda x: x.replace("none", "ncRNA"))
    
    df_int, df_neg = df[df.interacting == True], df[df.interacting == False]
    
    # PLOT 
    categories = list(set(df_int['where'].value_counts().index).union(df_neg['where'].value_counts().index))

    neg = df_neg['where'].value_counts()
    neg = pd.Series([neg.get(key, 0) for key in categories], index=categories)

    pos = df_int['where'].value_counts()
    pos = pd.Series([pos.get(key, 0) for key in categories], index=categories)

    values1 = pos.values

    values2 = neg.values

    total1 = sum(values1)
    total2 = sum(values2)

    percentages1 = np.array([value / total1 * 100 for value in values1])
    percentages2 = np.array([value / total2 * 100 for value in values2])

    bar_width = 0.35
    index = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=figsize)
    bar1 = ax.bar(index, percentages1, bar_width, label='Positive Distribution', color='skyblue')
    bar2 = ax.bar(index + bar_width, percentages2, bar_width, label='Negative Distribution', color='orange')

    #ax.set_xlabel('Categories')
    ax.set_ylabel('Percentage of regions pairs')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories, rotation=90, ha='right')
    if plot_legend:
        ax.legend()
    plt.tight_layout()
    
    if savepath:
        extension = find_extension_from_savepath(savepath)
        plt.savefig(savepath, format=f"{extension}", dpi = 300, bbox_inches='tight')
    
    plt.show()

def plot_correlations_QPC(corr_QPC, figsize=(5, 5), number_size = 7, savepath = ''):
    plt.figure(figsize=figsize)
    sns.heatmap(corr_QPC, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, annot_kws={"size": number_size})
    if savepath:
        extension = find_extension_from_savepath(savepath)
        plt.savefig(savepath, format=f"{extension}", dpi = 300, bbox_inches='tight')

def plot_qualityVSconfidence(n_reads, scores, variable='Number of Reads', figsize=(17, 9),  xlabel = 'xlabel', title = 'title', savepath = ''):
    """
    Plots a box plot for each list of scores corresponding to n_reads.
    
    Parameters:
    - n_reads: List of x-axis labels, representing different read numbers.
    - scores: List of lists, where each sublist contains the scores for the corresponding n_read.
    - variable: Label for the x-axis (default is 'Number of Reads').
    - figsize: Tuple to set the figure size (default is (17, 9)).
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Create the boxplot
    ax.boxplot(scores, showfliers=False)
    # Set the x-axis labels to the n_reads values
    ax.set_xticklabels(n_reads)
    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'{MODEL_NAME} Score')
    # Set the title
    ax.set_title(title)
    
    if savepath:
        extension = find_extension_from_savepath(savepath) 
        plt.savefig(savepath, format=f"{extension}", dpi = 300, bbox_inches='tight')
    
    # Display the plot
    plt.show()

# def plot_metric_confidence_for_all_models_for_2tasks(df_auc, list_of_reads, n_positives_run, name, size_multiplier = 10, metric = 'AUC', string_label = 'interaction length size', figsize = (17,9)):

#     model_names = list(df_auc['model_name'])

#     for task_name in ['interactors', 'patches']:
#         auc_models = []
#         perc_models = []
#         for model_name in model_names:
#             auc_current_model = []
#             for n_reads in list_of_reads:
#                 auc_current_model.append(df_auc[df_auc['model_name'] == model_name][f'auc_{task_name}_{name}{n_reads}'].iloc[0])
#             auc_models.append(auc_current_model)
#             perc_models.append(np.array(n_positives_run)/np.max(n_positives_run) * 100)

#         print('TASK: ', task_name)
#         plt.figure(figsize=figsize)
#         plot_metric_confidence_for_all_models(list_of_reads, auc_models, perc_models, model_names, task_name, size_multiplier, metric, string_label)
#         plt.show()
#         print('\n\n')

def plot_metric_confidence_for_all_models_for_2tasks(
    df_auc, list_of_reads, n_positives_run, name, titles, xlabel, ylabel,
    linewidth=2, legend_fontsize=5, size_multiplier=10, figsize=(17, 9), savepath=''
):
    model_names = list(df_auc['model_name'])

    for i, task_name in enumerate(['interactors', 'patches']):
        auc_models = []
        perc_models = []
        
        task_name_DR = 'DRP' if task_name == 'patches' else 'DRI'
        
        for model_name in model_names:
            auc_current_model = []
            for n_reads in list_of_reads:
                auc_current_model.append(
                    df_auc[df_auc['model_name'] == model_name][f'auc_{task_name_DR}_{name}{n_reads}'].iloc[0]
                )
            auc_models.append(auc_current_model)
            perc_models.append(np.array(n_positives_run) / np.max(n_positives_run) * 100)

        print('TASK: ', task_name)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the metrics
        plot_metric_confidence_for_all_models(
            list_of_reads, auc_models, perc_models, model_names, task_name, 
            size_multiplier, titles[i], xlabel, ylabel,
            second_xaxis=n_positives_run, second_xlabel = 'RRIs',
            linewidth=linewidth, legend_fontsize=legend_fontsize
        )

        # Layout adjustments and saving
        plt.tight_layout()
        
        if savepath:
            extension = find_extension_from_savepath(savepath)
            plt.savefig(add_task_name_to_savepath(savepath, task_name), format=f"{extension}", dpi=300, bbox_inches='tight')
        
        plt.show()
        print('\n\n')

def plot_metric_confidence_for_all_models(confidence_level, auc_models, perc_models, model_names, task_name, size_multiplier, title, xlabel, ylabel, second_xaxis, second_xlabel = 'RRIs', linewidth = 2, legend_fontsize = 5, markersize = 1):
    
    """
    auc_models: list
    perc_models: list
    model_names : list
    """
    
    for i, model_name in enumerate(model_names):
        model_color = model_colors_dict.get(model_name, 'black')
        
        plt.plot(confidence_level, auc_models[i], marker='o', markersize=markersize, label=map_model_names(model_name), color = model_color, linewidth=linewidth)
        #print(model_name, perc_models[i])
        # Opzionalmente, variare la dimensione dei punti in base alla numerosità
        for _, size in enumerate(perc_models[i]):
            plt.scatter(confidence_level[_], auc_models[i][_], s=float(size)*size_multiplier, color=model_color) #float(size)*size_multiplier
            
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_fontsize>0:
        plt.legend(fontsize = legend_fontsize)
    plt.grid(False)

    
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(confidence_level)  # Forza i tick dell'asse x a essere solo i valori interi

    # Ensure the ticks for the secondary axis are aligned with the primary axis
    secax = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))  # Keep the same mapping
    secax.set_xticks(confidence_level)  # Align the ticks with the primary axis
    secax.set_xticklabels(second_xaxis)  # Use second_xaxis as the labels
    secax.set_xlabel(second_xlabel)  # Set the label for the secondary x-axis

def plot_all_model_auc(subset, tools, n_runs=100, linewidth=2, legend_fontsize=6, figsize = (5,5), savepath = ''):
    models = [{'prob': subset.probability, 'model_name': 'NT'}]
    
    for tool_name in tools:
        models.append({'prob': abs(subset[tool_name]), 'model_name': tool_name})
    
    plot_roc_curves_with_undersampling(models, subset.ground_truth, n_runs = n_runs, linewidth=linewidth, legend_fontsize=legend_fontsize, figsize=figsize, savepath = savepath) 

def plot_roc_curves_with_undersampling(models, ground_truth, n_runs=50, linewidth=2, legend_fontsize=6, figsize=(5,5), savepath = ''):
    unbalanced = is_unbalanced(pd.DataFrame({'ground_truth': ground_truth}))
    
    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=linewidth) # Plotting the random guessing line

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
        
        plt.plot(mean_fpr, mean_tpr, label=f'{map_model_names(model["model_name"])} (AUC = {mean_auc:.2f})', color = model_colors_dict.get(model["model_name"], 'black'), linewidth=linewidth)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right', fontsize = legend_fontsize)
    
    if savepath:
        extension = find_extension_from_savepath(savepath) 
        plt.savefig(savepath, format=f"{extension}", dpi = 300)
    
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
    elif metric == 'cross_entropy':
        output = weighted_ce_loss(df, column, (1.0, 1.0))

    elif metric == 'cross_entropy_FP':
        output = weighted_ce_loss(df, column, (1.5, 1.0))

    elif metric == 'cross_entropy_FN':
        output = weighted_ce_loss(df, column, (1.0, 1.5))

    else:
        raise NotImplementedError

    return output


def weighted_ce_loss(subset, column, weights):
    """
    Compute weighted cross-entropy loss.

    Args:
        subset (pd.DataFrame): Dataframe containing predictions and ground truth.
        column (str): Name of the column containing predictions.
        weights (tuple): Weights for the negative (class 0) and positive (class 1) classes.

    Returns:
        float: Weighted cross-entropy loss.
    """
    # Clip predictions to avoid log(0)
    predictions = np.clip(subset[column], 1e-7, 1 - 1e-7)
    ground_truth = subset['ground_truth']
    
    # Separate weights for each class
    weight_0, weight_1 = weights

    # Compute weighted cross-entropy
    loss = -np.mean(
        ground_truth * weight_1 * np.log(predictions) +
        (1 - ground_truth) * weight_0 * np.log(1 - predictions)
    )
    return loss


def ce_loss(subset, column):
    return weighted_ce_loss(subset, column, (1.0, 1.0))


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
    
    plot_metric_confidence_for_all_models(confidence_level, auc_models, perc_models, model_names, task_name, size_multiplier, title = 'title', xlabel = 'xlabel', ylabel ='ylabel', second_xaxis = confidence_level, second_xlabel = 'duplicated_axis')
    
def collect_results_based_on_topbottom_for_all_models_perc_neg(df, MIN_PERC, list_of_models_to_test, n_values = 12, n_run_undersampling = 15, metric = 'precision'):
    
    auc_runs = []
    perc_runs = []
    
    perc_sn = []
    perc_hn = []
    perc_en = []

    
    for run in range(n_run_undersampling):
        
        subset = undersample_df(df) #undersampling at each run
        
        auc_models = []
        perc_models = []
        model_names = []

        perc_sn_run = []
        perc_hn_run = []
        perc_en_run = []

        confidence_level = []

        for model_name in list_of_models_to_test:
            
            percentages, metric_model, sn, hn, en = collect_results_based_on_confidence_level_based_on_topbottom_for_single_model_with_perc_neg(
                subset, how = model_name, n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric
            )
            
            auc_models.append(metric_model)
            perc_models.append(percentages)
            model_names.append(model_name)
            
            perc_sn_run.append(sn)
            perc_hn_run.append(hn)
            perc_en_run.append(en)

        if len(list_of_models_to_test)>1:
            assert perc_models[0] == perc_models[1]
        
        auc_runs.append(auc_models)
        perc_runs.append(perc_models)
        
        perc_sn.append(perc_sn_run)
        perc_hn.append(perc_hn_run)
        perc_en.append(perc_en_run)
    
    auc_models=np.mean(auc_runs, axis = 0)
    perc_models=np.mean(perc_runs, axis = 0)
    
    perc_sn=np.mean(perc_sn, axis = 0)
    perc_hn=np.mean(perc_hn, axis = 0)
    perc_en=np.mean(perc_en, axis = 0)
    
    return auc_models, perc_models, model_names, perc_sn, perc_hn, perc_en

def calc_perc_en_hn_sn(df):
    if df.shape[0]>0:
        perc_sn = df[df.policy == 'smartneg'].shape[0]/df.shape[0]
        perc_en = df[df.policy == 'easyneg'].shape[0]/df.shape[0]
        perc_hn = df[df.policy == 'hardneg'].shape[0]/df.shape[0]
    else: 
        perc_sn = 0
        perc_en = 0
        perc_hn = 0
    return perc_sn, perc_hn, perc_en

def collect_results_based_on_confidence_level_based_on_topbottom_for_single_model_with_perc_neg(df, how='nt', n_values=15, n_run_undersampling=30, MIN_PERC=0.05, metric='precision'):
    
    metric_model = []
    percentages = []
    sn = []
    hn = []
    en = []
    
    n_total = df.shape[0]
    
    percs_data = np.linspace(MIN_PERC, 100, n_values)[::-1]

    if how == 'nt':
        column = 'probability'
    else:
        column = how
        
    pred_pos = df[df[column] > 0.5].reset_index(drop=True)
    pred_neg = df[df[column] < 0.5].reset_index(drop=True)
    
    for i in range(n_values):
        n_to_sample = int(math.ceil(percs_data[i] / 100 * pred_pos.shape[0]))
        
        if metric == 'precision':
            subset = pred_pos.sort_values(column, ascending=False).head(n_to_sample)
            result_metric = calc_metric(subset, column, metric=metric)
            perc_sn, perc_hn, perc_en = calc_perc_en_hn_sn(subset)
            
        elif metric == 'npv':
            subset = pred_neg.sort_values(column, ascending=False).tail(n_to_sample)
            result_metric = calc_metric(subset, column, metric=metric)
            perc_sn, perc_hn, perc_en = calc_perc_en_hn_sn(subset)
        
        metric_model.append(result_metric)
        percentages.append(np.round(percs_data[i], 2))
        sn.append(perc_sn)
        hn.append(perc_hn)
        en.append(perc_en)
        
    return percentages, metric_model, sn, hn, en

def perc_neg_npv_precision(perc_sn_prec, perc_hn_prec, perc_en_prec, perc_sn_npv, perc_hn_npv, perc_en_npv, model_names, model_name, figsize, min_perc = 1):
    
    idx = int(np.where(np.array(model_names) == model_name)[0])
    model_names = map_model_names(model_names)
    
    perc_sn_prec = perc_sn_prec[idx]
    perc_hn_prec = perc_hn_prec[idx]
    perc_en_prec = perc_en_prec[idx]
    
    perc_sn_npv = perc_sn_npv[idx]
    perc_hn_npv = perc_hn_npv[idx]
    perc_en_npv = perc_en_npv[idx]
    
    n_points =  perc_sn_prec.shape[0]
    
    percentuali_neg = np.linspace(min_perc, 100, num=n_points).astype(int)
    percentuali_pos = np.linspace(min_perc, 100, num=n_points).astype(int)[::-1]

    percentuali = np.concatenate((percentuali_neg, percentuali_pos))
    
    combined_sn = np.hstack((perc_sn_npv, perc_sn_prec))
    combined_hn = np.hstack((perc_hn_npv, perc_hn_prec))
    combined_en = np.hstack((perc_en_npv, perc_en_prec))           

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    x_axis = np.arange(len(percentuali))
    
    plt.plot(x_axis, combined_sn, label = 'smartneg', linewidth=2)
    plt.plot(x_axis, combined_en, label = 'easyneg', linewidth=2)
    plt.plot(x_axis, combined_hn, label = 'hardneg', linewidth=2)        
    
    plt.axvline(x=n_points-0.5, color='black', linestyle='--', label='Threshold between positive and negative predictions')
    
    plt.xlabel('Percentage of bottom / top predictions (%)')
    plt.ylabel('Percentage of negatives')
    plt.title(f'Percentage of negatives in bottom predictions (left), and in top predictions (right) for model {map_model_names(model_name)}')
    plt.xticks(x_axis, percentuali)
                       
    plt.legend()

    plt.tight_layout()
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

    
def collect_results_based_on_topbottom_for_all_models(df, MIN_PERC, list_of_models_to_test, n_values = 12, n_run_undersampling = 15, metric = 'precision'):
    
    
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

        if len(list_of_models_to_test)>1:
            assert perc_models[0] == perc_models[1]
        
        auc_runs.append(auc_models)
        perc_runs.append(perc_models)
    
    auc_models=np.mean(auc_runs, axis = 0)
    perc_models=np.mean(perc_runs, axis = 0)
    
    return auc_models, perc_models, model_names
    
def plot_results_based_on_topbottom_for_all_models(df, MIN_PERC, list_of_models_to_test, n_values = 12, n_run_undersampling = 15, metric = 'precision', task_name = 'patches', size_multiplier = 10):
    
    
    auc_models, perc_models, model_names = collect_results_based_on_topbottom_for_all_models(df, MIN_PERC, list_of_models_to_test, n_values, n_run_undersampling, metric)

    plot_metric_confidence_for_all_models([str(perc) for perc in perc_models[0]], auc_models, perc_models, model_names, task_name, size_multiplier, title = 'title', xlabel = 'xlabel', ylabel ='ylabel',  second_xaxis = [str(perc) for perc in perc_models[0]], second_xlabel = 'duplicated_axis')

    
    
def get_results_based_on_topbottom_for_all_models(subset, MIN_PERC, how = 'nt', n_values = 12, n_run_undersampling = 30, metric = 'precision'):

    percentages, metric_model = collect_results_based_on_confidence_level_based_on_topbottom_for_single_model(subset, how = how, n_values = n_values, n_run_undersampling = n_run_undersampling, MIN_PERC = MIN_PERC, metric = metric)
    
    #perc_nt, perc_intarna, perc_ens = list(map(list, zip(*percentages)))
    
    return metric_model, percentages
    
    
    
def collect_results_based_on_confidence_level_based_on_topbottom_for_single_model(df, how='intarna', n_values=15, n_run_undersampling=30, MIN_PERC=0.05, metric='precision'):
    
    metric_model = []
    percentages = []
    
    n_total = df.shape[0]
    
    percs_data = np.linspace(MIN_PERC, 100, n_values)[::-1]

    # Determine the correct column based on the 'how' parameter
    if how == 'intarna':
        column = 'E_norm_conf'
    elif how == 'nt':
        column = 'probability'
    elif how == 'ensemble':
        column = 'ensemble_score'
    else:
        column = how
    
    # Ensure the column exists in the dataframe
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in the dataframe")
        
    pred_pos = df[df[column] > 0.5].reset_index(drop=True)
    pred_neg = df[df[column] < 0.5].reset_index(drop=True)
    
    for i in range(n_values):
        n_to_sample = int(math.ceil(percs_data[i] / 100 * pred_pos.shape[0]))
        
        if metric == 'precision':
            subset = pred_pos.sort_values(column, ascending=False).head(n_to_sample)
            result_metric = calc_metric(subset, column, metric=metric)
            
        elif metric == 'npv':
            subset = pred_neg.sort_values(column, ascending=False).tail(n_to_sample)
            result_metric = calc_metric(subset, column, metric=metric)
            
        elif metric in ['f1', 'recall', 'specificity', 'precision_recall_curve']:
            subset = pd.concat([
                pred_pos.sort_values(column, ascending=False).head(n_to_sample),
                pred_neg.sort_values(column, ascending=False).tail(n_to_sample)
            ], axis=0).reset_index(drop=True)
            result_metric = calc_metric(subset, column, metric=metric)
            
        elif metric == 'TP':
            subset = pred_pos.sort_values(column, ascending=False).head(n_to_sample)
            result_metric = subset[subset.ground_truth == 1].shape[0]
            
        elif metric == 'TN':
            subset = pred_neg.sort_values(column, ascending=False).tail(n_to_sample)
            result_metric = subset[subset.ground_truth == 0].shape[0]
            
        elif metric == 'FP':
            subset = pred_pos.sort_values(column, ascending=False).head(n_to_sample)
            result_metric = subset[subset.ground_truth == 0].shape[0]
            
        elif metric == 'FN':
            subset = pred_neg.sort_values(column, ascending=False).tail(n_to_sample)
            result_metric = subset[subset.ground_truth == 1].shape[0]
        
        metric_model.append(result_metric)
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
    
    plot_metric_confidence_for_all_models(confidence_level, perc_models, [[0 for i in range(len(perc_models[0]))] for j in range(len(perc_models))], model_names, feature_to_search, 0, title = 'title', xlabel = 'xlabel', ylabel ='ylabel',  second_xaxis = confidence_level, second_xlabel = 'duplicated_axis')

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
            tnr = calculate_tnr(subset, model)
            models_for_this_bin.append(tnr)
        models_tnr.append(models_for_this_bin)
   

    models_tnr = list(map(list, zip(*models_tnr)))
    
    plt.figure(figsize=figsize)
    plt.title('TNR of models in the task pathces based on distance interval')
    for i, model_name in enumerate(list_of_models_to_test):
        model_color = model_colors_dict.get(model_name, 'black')
        plt.plot(distances_axis, models_tnr[i], label = map_model_names(model_name), color = model_color, linewidth=2)

        for j, size in enumerate(percs):
            plt.scatter(distances_axis[j], models_tnr[i][j], s=float(size)*size_multiplier, color=model_color)

    plt.xlabel(f"Distance Interval")
    plt.ylabel(f"TNR Patches task")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
    
    
def calculate_tnr(subset, model):
    column = model if model != 'nt' else 'probability'
    
    # Filter for negative ground truth
    subset = subset[subset.ground_truth == 0]
    
    # Count the number of True Negatives and False Positives
    vc = (subset[column] < 0.5).value_counts()
    
    # Ensure vc[True] and vc[False] exist, handle possible edge cases
    true_negatives = vc.get(True, 0)
    false_positives = vc.get(False, 0)
    
    # Avoid division by zero
    if true_negatives + false_positives == 0:
        return None  # or return 0, depending on your use case
    
    # Compute TNR
    tnr = true_negatives / (true_negatives + false_positives)
    
    return tnr


def calculate_recall(subset, model):
    column = model if model != 'nt' else 'probability'
    
    # Count true positives (ground_truth is 1 and prediction > 0.5)
    true_positives = ((subset.ground_truth == 1) & (subset[column] > 0.5)).sum()
    
    # Count actual positives (ground_truth is 1)
    actual_positives = (subset.ground_truth == 1).sum()
    
    # Calculate recall and handle the case of zero actual positives
    recall = true_positives / actual_positives if actual_positives > 0 else None  # Could return 0 instead of None
    
    return recall

def plot_metrics_for_all_models(metric_function, ylabel, list_of_models_to_test, subset, figsize, title_suffix='', bar_width=0.5):
    metric_values = [metric_function(subset, model) for model in list_of_models_to_test]

    plt.figure(figsize=figsize)
    for i, (model, value) in enumerate(zip(list_of_models_to_test, metric_values)):
        model_color = model_colors_dict.get(model, 'black')
        plt.bar(i, value, width=bar_width, label=map_model_names(model), color=model_color)
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

    plt.xlabel('Models')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} for Different Models, {title_suffix}')
    plt.xticks(range(len(list_of_models_to_test)), list_of_models_to_test)
    plt.legend()
    plt.show()

def plot_tnr_for_all_models(list_of_models_to_test, subset, figsize, title_suffix='', bar_width=0.5):
    plot_metrics_for_all_models(calculate_tnr, 'TNR Values', list_of_models_to_test, subset, figsize, title_suffix, bar_width)

def plot_recall_for_all_models(list_of_models_to_test, subset, figsize, title_suffix='', bar_width=0.5):
    plot_metrics_for_all_models(calculate_recall, 'Recall Values', list_of_models_to_test, subset, figsize, title_suffix, bar_width)

    
def autolabel(ax, bars):
    """Attach a text label above each bar in *bars*, displaying its height."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_tnr_recall_for_all_models(list_of_models_to_test, subset, figsize, bar_width=0.35):
    tnr = [calculate_tnr(subset, model) for model in list_of_models_to_test]
    recall = [calculate_recall(subset, model) for model in list_of_models_to_test]

    list_of_models_to_test = map_model_names(list_of_models_to_test)
    
    x = np.arange(len(list_of_models_to_test))  # the label locations

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - bar_width/2, tnr, bar_width, label='TNR', color='skyblue')
    bars2 = ax.bar(x + bar_width/2, recall, bar_width, label='Recall', color='lightcoral')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('TNR and Recall for Different Models')
    ax.set_xticks(x)
    ax.set_xticklabels(list_of_models_to_test)
    ax.legend()

    autolabel(ax, bars1)
    autolabel(ax, bars2)

    fig.tight_layout()
    plt.show()

def plot_tnr_based_on_distance_for_our_model(ephnen, enhn500, bins_distance, figsize, size_multiplier):
    
    plt.figure(figsize=figsize)
    plt.title('TNR of models in the task pathces based on distance interval')
    
    for _, enhn in enumerate([ephnen, enhn500]):
        
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
            distances_axis.append(str([dist1, dist2]))

            models_for_this_bin = []
            column = 'probability'
            tnr = (subset.ground_truth == (subset[column] > 0.5).astype(int)).sum() / subset.shape[0]
            models_for_this_bin.append(tnr)
            models_tnr.append(models_for_this_bin)


        models_tnr = list(map(list, zip(*models_tnr)))

        model_name = 'nt'

        model_color = model_colors_dict.get(model_name, 'black')
        
        label = 'Interacting region present within embedding' if _ == 0 else 'No interacting region within embedding'
        linestyle = '--' if _ == 0 else 'dotted'
        
        plt.plot(distances_axis, models_tnr[0], label = label, color = model_color, linewidth=2, linestyle = linestyle)
        
        for j, size in enumerate(percs):
            plt.scatter(distances_axis[j], models_tnr[0][j], s=float(size)*size_multiplier, color=model_color)

        plt.xlabel(f"Distance Interval")
        plt.ylabel(f"TNR Patches task")
        plt.legend()
        plt.grid(True, alpha=0.5)
    plt.show()
    
def plot_trn_recall_for_all_models(list_of_models_to_test, subset, figsize):

    tnr = [calculate_tnr(subset, model) for model in list_of_models_to_test]
    recall = [calculate_recall(subset, model) for model in list_of_models_to_test]

    x = np.arange(len(list_of_models_to_test))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, tnr, width, label='TNR', color='skyblue')
    bars2 = ax.bar(x + width/2, recall, width, label='Recall', color='lightcoral')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('TNR and Recall for Different Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Attach a text label above each bar in *bars*, displaying its height.
    def autolabel(bars):
        """Attach a text label above each bar in *bars*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    fig.tight_layout()

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
        model_color = model_colors_dict.get(model_name, 'black')
    
        plt.plot(distances_axis, models_conifdence[i], label = map_model_names(model_name), color = model_color, linewidth=2)
        
        regression_line = obtain_regression_line(models_conifdence[i])
        plt.plot(distances_axis, regression_line, color = model_color, linewidth=2, linestyle = '--')

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
    
    
def plot_heatmap(correlation_df, highlight_labels=None, title="Correlation Heatmap", cmap="coolwarm", annot=True, method='pearson', numbers_size = 5, figsize=(10, 8), savepath=''):
    """
    Plot a heatmap of the given correlation DataFrame.

    Parameters:
    - correlation_df: DataFrame containing the correlation matrix.
    - highlight_labels: List of labels to highlight on the heatmap.
    - title: Title of the heatmap.
    - cmap: Colormap to use for the heatmap.
    - annot: Boolean indicating whether to annotate the heatmap with correlation values.
    - method: Correlation method to use.
    - figsize: Tuple specifying the figure size.
    - savepath: Path to save the heatmap image.
    """
    # Map column names
    mapped_columns = map_model_names(correlation_df.columns.tolist())
    correlation_df.columns = mapped_columns
    correlation_df.index = mapped_columns

    # Create the clustermap
    clustergrid = sns.clustermap(
        correlation_df,
        annot=annot,
        cmap=cmap,
        vmin=-1 if method != 'mutual_info' else None,
        vmax=1 if method != 'mutual_info' else None,
        center=0 if method != 'mutual_info' else None,
        figsize=figsize,  # Pass figsize here
        method='average',  # Example clustering method
        metric='euclidean',  # Example distance metric
    )

    # Set the title for the heatmap
    #clustergrid.ax_heatmap.set_title(title, fontsize=plt.rcParams['font.size'])

    # Highlight specified labels
    if highlight_labels:
        ax = clustergrid.ax_heatmap
        labels = correlation_df.columns
        # for label in labels:
        #     if label in highlight_labels:
        #         idx = list(labels).index(label)
        #         ax.get_xticklabels()[idx].set_color('darkgreen')
        #         ax.get_yticklabels()[idx].set_color('darkgreen')

    #clustergrid.ax_heatmap.tick_params(axis='both', which='both', labelsize=numbers_size)
    clustergrid.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=6)

    
    # Save the figure if a savepath is provided
    if savepath:
        extension = find_extension_from_savepath(savepath)
        clustergrid.savefig(savepath, format=f"{extension}", dpi=300)

    plt.show()

    
def plot_sr_distributions_full(df_sr, label_x = 'Dataset', label_x_name = 'Dataset', label_y_name = 'Normalized Score', column = 'Model', title = 'title',linewidth_boxplot = 0.5, width_boxplot=0.7, figsize = (16, 8), savepath = ''):

    df_sr[column] = df_sr[column].apply(map_model_names)
    if 'Dataset' in df_sr.columns:
        df_sr['Dataset'] = df_sr['Dataset'].apply(map_experiment_names)
    
    plt.figure(figsize=figsize)

    hue_order = [
        'no Simple Repeat positive samples',
        'at least Simple Repeat positive samples',
        'both Simple Repeat positive samples',
    ]
    palette={hue_order[0]: '#D5E7B5', hue_order[1]: '#8174A0', hue_order[2]: '#441752'}

    ax = sns.boxplot(data=df_sr, x=label_x, y='Normalized Score', hue='Category', hue_order=hue_order, palette = palette, showfliers=False, width = width_boxplot, linewidth=linewidth_boxplot)
    #sns.violinplot(data=df_sr, x=label_x, y='Normalized Score', hue='Category', hue_order=hue_order, palette = palette, showfliers=False)
        
        
    # Aggiungi i segmenti orizzontali sopra i box plot
    for i in range(len(df_sr[label_x].unique())):  # Per ogni categoria sull'asse x
        x_positions = [i - 0.25, i, i + 0.25]  # Posizioni x dei box plot
        y_value = df_sr['Normalized Score'].max() + 0.05  # Altezza sopra i box plot
        segment_width = 0.2  # Ampiezza del segmento
        # Aggiungi segmento sopra il secondo box plot (tra box1 e box2)
        ax.hlines(y=y_value, xmin=(x_positions[0] + x_positions[1]) / 2 - segment_width / 2, xmax=(x_positions[0] + x_positions[1]) / 2 + segment_width / 2, color='black', linewidth=0.3)
        # Aggiungi segmento sopra il terzo box plot (tra box1 e box3)
        ax.hlines(y=y_value + 0.1, xmin=(x_positions[0] + x_positions[1]) / 2 - segment_width / 2, xmax=(x_positions[1] + x_positions[2]) / 2 + segment_width / 2, color='black', linewidth=0.3)
    
    # Customize the plot
    plt.title(title)
    plt.xlabel(label_x_name)
    plt.ylabel(label_y_name)
    plt.ylim(plt.ylim()[0], plt.ylim()[1] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05)
    #plt.legend(title='Dataset')
    plt.tight_layout()
    plt.legend([],[], frameon=False)

    if savepath:
        extension = find_extension_from_savepath(savepath) 
        plt.savefig(savepath, format=f"{extension}", dpi = 300)
    
    plt.show()
    
def plot_sr_distributions(df_sr, label_x, label_y_name = 'Normalized Score', column = 'Model', figsize = (16, 8), savepath = ''):
    
    #map model_names
    df_sr[column] = df_sr[column].apply(map_model_names)

    # Create the violin plot without the inner box plot
    plt.figure(figsize=figsize)
    ax = sns.violinplot(x=column, y='Normalized Score', hue='Category', data=df_sr, split=True, palette=['#FF9999', '#99FF99'], inner=None)

    # Add the mean points with custom horizontal lines
    mean_line_length = 0.3  # Adjust this value to control the length of the horizontal lines

    # Calculate means
    mean_points = df_sr.groupby([column, 'Category'])['Normalized Score'].mean().reset_index()

    # Get the positions of each category for plotting
    positions = {category: idx for idx, category in enumerate(df_sr[column].unique())}

    # Plot mean lines manually
    for i, model in enumerate(mean_points[column].unique()):
        for j, category in enumerate(mean_points['Category'].unique()):
            mean_val = mean_points[(mean_points[column] == model) & (mean_points['Category'] == category)]['Normalized Score'].values[0]
            pos = positions[model]
            # Add offset for split violin
            if category == label_x:
                pos -= mean_line_length / 2
            else:
                pos += mean_line_length / 2
            plt.plot([pos - mean_line_length / 2, pos + mean_line_length / 2], [mean_val, mean_val], color=['#A26565', '#568E56'][j], lw=2)

    # Adjust the legend to prevent duplication
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title='Category')
    plt.ylabel(label_y_name)
    plt.title('Violin Plot with Two Distributions per Category')
    
    if savepath:
        extension = find_extension_from_savepath(savepath) 
        plt.savefig(savepath, format=f"{extension}", dpi = 300)
    
    plt.show()
    
    
def npv_precision(precision_data, npv_data, experiment_names, figsize, min_perc = 1, title = 'NPV over bottom predictions (left), Precision over top predictions (right)', savepath = '', plot_legend = False):
    
    experiment_names = map_experiment_names(experiment_names)
    
    assert precision_data.shape == npv_data.shape
    
    num_modelli, n_points =  precision_data.shape[0], precision_data.shape[1]
    
    percentuali_neg = np.linspace(min_perc, 100, num=n_points).astype(int)
    percentuali_pos = np.linspace(min_perc, 100, num=n_points).astype(int)[::-1]

    percentuali = np.concatenate((percentuali_neg, percentuali_pos))
    
    # Unione dei dati di precisione e NPV in un unico array per l'asse Y
    combined_data = np.hstack((npv_data, precision_data))

    # Creiamo un array per la combinazione delle due colormap
    combined_image = np.zeros((num_modelli, 2*n_points, 3))

    # Applichiamo la colormap 'Oranges' alla parte sinistra (NPV)
    norm = plt.Normalize(vmin=0, vmax=1)
    npv_colored = plt.cm.Blues(norm(npv_data))[:, :, :3]  # Consideriamo solo i primi tre canali (RGB)
    combined_image[:, :n_points, :] = npv_colored

    # Applichiamo la colormap 'Blues' alla parte destra (Precision)
    precision_colored = plt.cm.Oranges(norm(precision_data))[:, :, :3]  # Consideriamo solo i primi tre canali (RGB)
    combined_image[:, n_points:, :] = precision_colored

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # Creiamo il grafico principale
    im = ax.imshow(combined_image, aspect='auto', interpolation='nearest')
    plt.xlabel('Percentage of bottom / top predictions (%)')
    #plt.ylabel('Model')
    plt.axvline(x=n_points-0.5, color='black', linewidth=0.5, linestyle='--', label='Threshold between positive and negative predictions')
    plt.title(title)
    plt.xticks(np.arange(len(percentuali)), percentuali)
    plt.yticks(np.arange(num_modelli), experiment_names)

    if plot_legend:

        # Divider per le due colorbar
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.1)
        cax2 = divider.append_axes("right", size="5%", pad=0.7)

        # Colorbar per NPV
        cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Blues'), cax=cax1)
        cb1.set_label('NPV Score')
        cax1.tick_params(axis='y', labelrotation=90, labelsize=7)  #size of the numbers

        # Colorbar per Precision
        cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Oranges'), cax=cax2)
        cb2.set_label('Precision Score')
        cax2.tick_params(axis='y', labelrotation=90, labelsize=7)  #size of the numbers

    # Aggiungiamo i numeri alle celle del grafico
    for i in range(num_modelli):
        for j in range(n_points):
            text = ax.text(j, i, f"{npv_data[i, j]:.2f}", ha="center", va="center", color="white",  fontsize=5, fontfamily='Arial')
            text = ax.text(j + n_points, i, f"{precision_data[i, j]:.2f}", ha="center", va="center", color="white", fontsize=5, fontfamily='Arial')

    plt.tight_layout()
    
    if savepath:
        extension = find_extension_from_savepath(savepath) 
        plt.savefig(savepath, format=f"{extension}", dpi = 300)
    
    plt.show()    
    
def plot_correlation_nreads_prob_intsize(modelRM, PARIS_FINETUNED_MODEL):

    res = modelRM.get_experiment_data(
        experiment = 'paris', 
        paris_test = True, 
        paris_finetuned_model = PARIS_FINETUNED_MODEL, 
        specie_paris = 'all',
        paris_hq = False,
        paris_hq_threshold = 1,
        n_reads_paris = 1,
        interlen_OR_nreads_paris = False,
        splash_trained_model = False,
        only_test_splash_ricseq_mario = np.nan,
        n_reads_ricseq = np.nan,
        n_reads_mario = np.nan,
        logistic_regression_models = {},
    )

    pos = res[res.interacting]


    # Calculate the mean size of interaction
    mean_size_interaction = np.log( ((pos.seed_x2 - pos.seed_x1) + (pos.seed_y2 - pos.seed_y1)) / 2 )

    # Calculate Pearson correlation coefficients
    corr_size_reads, _ = pearsonr(mean_size_interaction, pos.n_reads)
    corr_size_prob, _ = pearsonr(mean_size_interaction, pos.probability)
    corr_reads_prob, _ = pearsonr(pos.n_reads, pos.probability)

    # Display correlation coefficients
    print(f"Correlation between log(interaction size) and n_reads: {corr_size_reads:.2f}")
    print(f"Correlation between log(interaction size) and probability: {corr_size_prob:.2f}")
    print(f"Correlation between n_reads and probability: {corr_reads_prob:.2f}")

    # Set up a grid for subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot size vs n_reads
    sns.scatterplot(x=mean_size_interaction, y=pos.n_reads, ax=ax[0])
    sns.regplot(x=mean_size_interaction, y=pos.n_reads, ax=ax[0], scatter=False, color='red')
    ax[0].set_title(f"Size vs n_reads\nCorr: {corr_size_reads:.2f}")
    ax[0].set_xlabel('Log Mean Size Interaction')
    ax[0].set_ylabel('n_reads')

    # Plot size vs probability
    sns.scatterplot(x=mean_size_interaction, y=pos.probability, ax=ax[1])
    sns.regplot(x=mean_size_interaction, y=pos.probability, ax=ax[1], scatter=False, color='red')
    ax[1].set_title(f"Size vs Probability\nCorr: {corr_size_prob:.2f}")
    ax[1].set_xlabel('Log Mean Size Interaction')
    ax[1].set_ylabel('Probability')

    # Plot n_reads vs probability
    sns.scatterplot(x=pos.n_reads, y=pos.probability, ax=ax[2])
    sns.regplot(x=pos.n_reads, y=pos.probability, ax=ax[2], scatter=False, color='red')
    ax[2].set_title(f"n_reads vs Probability\nCorr: {corr_reads_prob:.2f}")
    ax[2].set_xlabel('n_reads')
    ax[2].set_ylabel('Probability')

    # Show plot
    plt.tight_layout()
    plt.show()