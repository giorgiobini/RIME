import pandas as pd
import numpy as np
import os
import seaborn as sns
import pickle
from tqdm.notebook import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
from .plot_utils import get_results_based_on_treshold
from .misc import balance_df

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *



'''
------------------------------------------------------------------------------------------
'''
# INTARNA MAP

def custom_sigmoid(x, slope_at_origin=2):
    return 0.5 + 0.5 * np.tanh(x * slope_at_origin)

def map_signal_to_sigmoid_range(signal, threshold):
    mapped_signal = custom_sigmoid((signal - threshold))
    return mapped_signal

'''
------------------------------------------------------------------------------------------
'''



def load_paris_results(checkpoint_dir, test500, HOW, SPECIE, show_plot_map_signal = False):

    file_train = os.path.join(rna_rna_files_dir, "gene_pairs_training.txt")
    with open(file_train, "rb") as fp:   # Unpickling
        gene_pairs_train_original = pickle.load(fp)

    file_train = os.path.join(rna_rna_files_dir, "gene_pairs_training_nt_HQ.txt")
    with open(file_train, "rb") as fp:   # Unpickling
        gene_pairs_train = pickle.load(fp)

    file_test = os.path.join(rna_rna_files_dir, f"gene_pairs_{HOW}_nt_HQ.txt")
    with open(file_test, "rb") as fp:   # Unpickling
        gene_pairs_test = pickle.load(fp)

    # file_test_subset = os.path.join(rna_rna_files_dir, f"gene_pairs_{HOW}_sampled_nt_HQ.txt")
    # with open(file_test_subset, "rb") as fp:   # Unpickling
    #     gene_pairs_test_subset = pickle.load(fp)
    
    assert test500.couples.isin(gene_pairs_test).all()
     
    #-------------- -------------- -------------- -------------- -------------- -------------- --------------
    
    res = pd.read_csv(os.path.join(checkpoint_dir, f'{HOW}_results500.csv'))

    # Drop all the pairs (they should be 60-70) that are present in the training set.
    res = res[~res.couples.isin(gene_pairs_train_original)]

    # show only results for 1 specie
    res = res[res.specie == SPECIE]

    #-------------- -------------- -------------- -------------- -------------- -------------- --------------
    
    intarna = pd.read_csv(os.path.join(intarna_dir, f'{HOW}500', f'{HOW}.csv'), sep = ';')
    intarna['key'] = intarna.id1 + '_' + intarna.id2

    # keep only the lower E_norm for each group
    intarna.sort_values('E_norm', ascending = False, inplace=True)
    intarna.drop_duplicates(subset='key', keep='first', inplace=True)
    intarna = intarna.reset_index(drop = True)
    intarna['couples'] = intarna.id1.str.extractall('(.*)_(.*)').reset_index(drop = True)[0]
    intarna['couples'] = intarna['couples'].astype(int)

    intarna = intarna.dropna()

    res = res.merge(intarna[['E','E_norm', 'couples']].rename({'couples':'id_sample'}, axis =1), on = 'id_sample')
    res['original_area'] = res.original_length1 * res.original_length2
    
    #-------------- -------------- -------------- -------------- -------------- -------------- --------------
    
    intarna_treshold = -1.25

    # Example usage
    signal_range = np.linspace(-5, 0, 1000)

    mapped_signal = map_signal_to_sigmoid_range(signal_range, intarna_treshold)

    if show_plot_map_signal:
        # Plotting the results
        plt.plot(signal_range, mapped_signal, label='Mapped Signal')
        plt.axvline(x=intarna_treshold, color='r', linestyle='--', label='Intarna Threshold to be 0.5')
        plt.axhline(y=0.5, color='g', linestyle='--', label='0.5')
        plt.xlabel('Original Signal')
        plt.ylabel('Mapped Signal')
        plt.legend()
        plt.show()

    # Ranking the 'E_norm' column in ascending order
    res['E_norm_conf'] = map_signal_to_sigmoid_range(res['E_norm'], intarna_treshold)
    res['E_norm_conf'] = 1 - res['E_norm_conf']

    ### L agreement score in questo caso di intarna non tiene conto dello sbilanciamento dei dati... dovrei prima trovare qual e l la soglia di INTARNA dove mettere lo 0.5 e poi fare la media con lo score del nostro modello
    res['ensemble_score'] = (res['probability'] + res['E_norm_conf']) / 2
    assert res.ensemble_score.max() <= 1
    assert res.ensemble_score.min() >= 0
    
    return res

def load_ricseq_splash_mario_results(checkpoint_dir, test500, df_nt, how, only_test, exclude_train_genes, exclude_paris_genes, exclude_paris_couples, filter_hq_ricseq, MIN_N_READS_RICSEQ, show_plot_map_signal = False):
    
    res = pd.read_csv(os.path.join(checkpoint_dir, f'{how}_results500.csv'))
    
    if (how == 'ricseq')&filter_hq_ricseq:
        ricsechq = pd.read_csv(os.path.join(os.path.join('ricseqHQ_couples.csv')))
        hq = res[res.couples.isin(ricsechq['couples'])]
        pos_to_keep = list(hq[hq.policy == 'easypos'].id_sample)
        smartneg_to_drop = list(hq[hq.policy == 'smartneg'].id_sample)
        res = res[
            (res.id_sample.isin(pos_to_keep)) | 
            ((res.policy == 'smartneg') & (~res.id_sample.isin(smartneg_to_drop)) ) | 
            ((res.policy == 'hardneg') & (res.id_sample.isin(pos_to_keep)) ) |
            ((res.policy == 'easyneg') & (~res.id_sample.isin(smartneg_to_drop)) )
        ]
        
    #-------------- -------------- -------------- -------------- -------------- -------------- --------------
        
    file_train = os.path.join(rna_rna_files_dir, f'{how}', 'gene_pairs_training.txt')
    with open(file_train, "rb") as fp:   # Unpickling
        train_couples = pickle.load(fp)

    file_test = os.path.join(rna_rna_files_dir, f'{how}', 'gene_pairs_test.txt')
    with open(file_test, "rb") as fp:   # Unpickling
        test_couples = pickle.load(fp)

    train_paris = os.path.join(rna_rna_files_dir, 'gene_pairs_training.txt')
    with open(train_paris, "rb") as fp:   # Unpickling
        paris_couples1 = pickle.load(fp)
    val_paris = os.path.join(rna_rna_files_dir, 'gene_pairs_val.txt')
    with open(val_paris, "rb") as fp:   # Unpickling
        paris_couples2 = pickle.load(fp)

    paris_couples=pd.Series(list(set(paris_couples1).union(paris_couples2))).str.extractall('(.*)_(.*)').reset_index()
    paris_genes = set(paris_couples[0]).union(paris_couples[1])    
    paris_couples = set(paris_couples1).union(paris_couples2)


    tr_genes=pd.Series(train_couples).str.extractall('(.*)_(.*)').reset_index()
    training_genes = set(tr_genes[0]).union(tr_genes[1])

    if only_test:
        res = res[res.couples.isin(test_couples)]
        if exclude_train_genes:
            res = res[~(res.gene1_original.isin(training_genes) | res.gene2_original.isin(training_genes))]

    if exclude_paris_genes:
        n_original_couples=res.shape[0]
        res = res[~(res.gene1_original.isin(paris_genes) | res.gene2_original.isin(paris_genes))]
        print('# excluded couples: ', n_original_couples - res.shape[0])
    if exclude_paris_couples:
        n_original_couples=res.shape[0]
        res = res[~(res.couples.isin(paris_couples))]
        print('# excluded couples: ', n_original_couples - res.shape[0])
        
    #-------------- -------------- -------------- -------------- -------------- -------------- --------------

    assert test500.shape[0] == df_nt[['couples', 'interacting']].merge(test500, on = 'couples').shape[0]

    if how == 'ricseq':
        test500 = df_nt[['couples', 'interacting', 'where', 'where_x1', 'where_y1', 'simple_repeats', 'sine_alu', 'low_complex', 'n_reads']].merge(test500, on = 'couples')
        ids_to_keep = set(test500[test500.n_reads >= MIN_N_READS_RICSEQ].couples).union(test500[test500.interacting==False].couples)
        res = res[res.id_sample.isin(ids_to_keep)]
    elif how == 'mario':
        test500 = df_nt[['couples', 'interacting', 'where', 'where_x1', 'where_y1', 'simple_repeats', 'sine_alu', 'low_complex', 'n_reads']].merge(test500, on = 'couples')
    elif how == 'splash':
        test500 = df_nt[['couples', 'interacting', 'where', 'where_x1', 'where_y1', 'experiment']].merge(test500, on = 'couples')
    else:
        raise NotImplementedError

    id_cds_cds = set(test500[test500['where'] == 'CDS-CDS'].couples)
    
    #-------------- -------------- -------------- -------------- -------------- -------------- --------------
    
    intarna = pd.read_csv(os.path.join(intarna_dir, f'{how}500_RANDOM', f'{how}.csv'), sep = ';')
    intarna['key'] = intarna.id1 + '_' + intarna.id2

    # keep only the lower E_norm for each group
    intarna.sort_values('E_norm', ascending = False, inplace=True)
    intarna.drop_duplicates(subset='key', keep='first', inplace=True)
    intarna = intarna.reset_index(drop = True)
    intarna['couples'] = intarna.id1.str.extractall('(.*)_(.*)').reset_index(drop = True)[0]
    intarna['couples'] = intarna['couples'].astype(int)

    intarna = intarna.dropna()


    res = res.merge(intarna[['E','E_norm', 'couples']].rename({'couples':'id_sample'}, axis =1), on = 'id_sample')
    res['original_area'] = res.original_length1 * res.original_length2
    
    
    #-------------- -------------- -------------- -------------- -------------- -------------- --------------
    
    intarna_treshold = -1.25

    # Example usage
    signal_range = np.linspace(-5, 0, 1000)

    mapped_signal = map_signal_to_sigmoid_range(signal_range, intarna_treshold)
    
    if show_plot_map_signal:
        # Plotting the results
        plt.plot(signal_range, mapped_signal, label='Mapped Signal')
        plt.axvline(x=intarna_treshold, color='r', linestyle='--', label='Intarna Threshold to be 0.5')
        plt.axhline(y=0.5, color='g', linestyle='--', label='0.5')
        plt.xlabel('Original Signal')
        plt.ylabel('Mapped Signal')
        plt.legend()
        plt.show()

    # Ranking the 'E_norm' column in ascending order
    res['E_norm_conf'] = map_signal_to_sigmoid_range(res['E_norm'], intarna_treshold)
    res['E_norm_conf'] = 1 - res['E_norm_conf']

    ### L agreement score in questo caso di intarna non tiene conto dello sbilanciamento dei dati... dovrei prima trovare qual e l la soglia di INTARNA dove mettere lo 0.5 e poi fare la media con lo score del nostro modello
    res['ensemble_score'] = (res['probability'] + res['E_norm_conf']) / 2
    assert res.ensemble_score.max() <= 1
    assert res.ensemble_score.min() >= 0
    
    return res



def load_results(checkpoint_dir_paths, space, n_values, MIN_PERC, chkpt_folder):
    
    filename_output = os.path.join(chkpt_folder, 'model_performance.pkl')
    
    if os.path.exists(filename_output):
        # Loading the variables from the file
        with open(filename_output, 'rb') as file:
            df_full, name_map, confidence_level = pickle.load(file)
        
        to_do = list( set(checkpoint_dir_paths) - set(name_map.values()) )
        
        if len(to_do) > 0:
        
            new_df_full, new_name_map, new_confidence_level = otain_results(to_do, space, n_values, MIN_PERC, index_name = len(name_map))

            df_full = pd.concat([df_full,new_df_full], axis = 0).reset_index(drop = True)
            name_map.update(new_name_map)

            # Saving the variables to a file
            with open(filename_output, 'wb') as file:
                pickle.dump((df_full, name_map, confidence_level), file)

    else:
        df_full, name_map, confidence_level = otain_results(checkpoint_dir_paths, space, n_values, MIN_PERC)
        # Saving the variables to a file
        with open(filename_output, 'wb') as file:
            pickle.dump((df_full, name_map, confidence_level), file)
        
    return df_full, name_map, confidence_level


def obtain_auc_nt_intarna(res):
    fpr, tpr, _ = roc_curve(res.ground_truth, res.probability)
    roc_auc_nt = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(res.ground_truth, abs(res.E_norm))
    roc_auc_intarna = auc(fpr, tpr)
    return roc_auc_nt, roc_auc_intarna

def add_row_values(task_name, set_name, subset, row, n_values, n_run_undersampling, MIN_PERC):
    for metric in ['precision_recall_curve', 'f1']:
        # Get results based on threshold for NT
        confidence_level, metric_nt, perc_nt, _, _, _, _ = get_results_based_on_treshold(subset=subset, MIN_PERC=MIN_PERC, n_values=n_values, order_by='nt', n_run_undersampling=n_run_undersampling, metric=metric, consensus=False)
        
        # Add NT confidence level results to row
        for i in range(len(confidence_level)):
            row[f'{set_name}_{task_name}_{metric}_NTconf{confidence_level[i]}'] = metric_nt[i]

        # Get results based on threshold for INTARNA
        _, _, _, metric_intarna, perc_intarna, _, _ = get_results_based_on_treshold(subset=subset, MIN_PERC=MIN_PERC, n_values=n_values, order_by='intarna', n_run_undersampling=n_run_undersampling, metric=metric, consensus=False)
        
        # Add INTARNA confidence level results to row
        for i in range(len(confidence_level)):
            row[f'{set_name}_{task_name}_{metric}_INTARNAconf{confidence_level[i]}'] = metric_intarna[i]
            
    fpr, tpr, _ = roc_curve(subset.ground_truth, subset.probability)
    roc_auc = auc(fpr, tpr)
    row[f'{set_name}_auc_NT_{task_name}'] = roc_auc

    fpr, tpr, _ = roc_curve(abs(1 - subset.ground_truth), subset.E_norm)
    roc_auc = auc(fpr, tpr)
    row[f'{set_name}_auc_INTARNA_{task_name}'] = roc_auc
    
    return row, confidence_level

def otain_results(checkpoint_dir_paths, space, n_values, MIN_PERC, index_name = 0):

    diz_results = {}
    name_map = {}
    
    for idx_v, checkpoint_dir in tqdm(enumerate(checkpoint_dir_paths), total = len(checkpoint_dir_paths)):
        row = {}

        # -----------
        # PARIS args
        # -----------
        
        HOW = 'test'
        SPECIE = 'human'
        
        # -----------------------
        # RICSEQ and SPLASH args
        # -----------------------

        only_test = False
        exclude_train_genes = False
        exclude_paris_genes = False
        exclude_paris_couples = True

        filter_hq_ricseq = False

        MIN_N_READS_RICSEQ = 3
        
        # -----------
        # Calculation
        # ------------
        
        n_run_undersampling = 30
        n_values = 12
        MIN_PERC = 1

        for how in ['paris', 'ricseq', 'splash']:
            
            if how == 'paris':
                test500 = pd.read_csv(os.path.join(metadata_dir, f'{HOW}500.csv'))
                df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_HQ.csv'))
                assert test500.shape[0] == df_nt[['couples', 'interacting']].merge(test500, on = 'couples').shape[0]
                test500 = df_nt[['couples', 'interacting', 'where', 'where_x1', 'where_y1', 'simple_repeats', 'sine_alu', 'low_complex']].merge(test500, on = 'couples')
                id_cds_cds = set(test500[test500['where'] == 'CDS-CDS'].couples)
                res = load_paris_results(checkpoint_dir, test500, HOW, SPECIE)

            else:
                test500 = pd.read_csv(os.path.join(metadata_dir, f'{how}500.csv'))
                df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_{how}.csv'))
                res = load_ricseq_splash_mario_results(checkpoint_dir, test500, df_nt, how, only_test, exclude_train_genes, exclude_paris_genes, exclude_paris_couples, filter_hq_ricseq, MIN_N_READS_RICSEQ)

            easypos_smartneg = res[res.policy.isin(['smartneg', 'easypos'])].reset_index(drop = True)
            subset = easypos_smartneg.copy()
            subset = balance_df(subset).reset_index(drop = True)
            row, confidence_level = add_row_values('interactors', how, subset, row, n_values, n_run_undersampling, MIN_PERC)
            
            ephnen = res[(res.policy == 'easyneg')|(res.policy == 'easypos')|(res.policy == 'hardneg')].reset_index(drop = True)
            subset = ephnen.copy()
            subset = balance_df(subset).reset_index(drop = True)
            row, _ = add_row_values('patches', how, subset, row, n_values, n_run_undersampling, MIN_PERC)
            
        # -------------------
        # SPLASH RICSEQ test
        # -------------------
        only_test = True
        for how in ['ricseq', 'splash']:
            test500 = pd.read_csv(os.path.join(metadata_dir, f'{how}500.csv'))
            df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_{how}.csv'))
            res = load_ricseq_splash_mario_results(checkpoint_dir, test500, df_nt, how, only_test, exclude_train_genes, exclude_paris_genes, exclude_paris_couples, filter_hq_ricseq, MIN_N_READS_RICSEQ)
            easypos_smartneg = res[res.policy.isin(['smartneg', 'easypos'])].reset_index(drop = True)
            subset = easypos_smartneg.copy()
            subset = balance_df(subset).reset_index(drop = True)
            row, _ = add_row_values('interactors', how+'test', subset, row, n_values, n_run_undersampling, MIN_PERC)
            
            ephnen = res[(res.policy == 'easyneg')|(res.policy == 'easypos')|(res.policy == 'hardneg')].reset_index(drop = True)
            subset = ephnen.copy()
            subset = balance_df(subset).reset_index(drop = True)
            row, _ = add_row_values('patches', how+'test', subset, row, n_values, n_run_undersampling, MIN_PERC)
            
        model_name = f'model{idx_v + index_name}'
        name_map[model_name] = checkpoint_dir
        diz_results[model_name] = row
    
    df = pd.DataFrame.from_dict(diz_results, 'index')
    df = df*100
    df = df.round(2)
    df = df.reset_index().rename({'index':'model'}, axis = 1)
    
    intarna_columns = []
    all_columns = list(df.columns)
    for col in all_columns:
        if 'INTARNA' in col:
            intarna_columns.append(col)


    df_intarna, other = df.filter(intarna_columns, axis = 1), df.filter(set(df.columns) - set(intarna_columns), axis = 1)

    map_name = {}
    for col in df_intarna.columns:
        if 'INTARNAconf' in col:
            col_clean = col.replace('INTARNAconf', 'NTconf')
            map_name[col] = col_clean
        elif 'INTARNA_' in col:
            col_clean = col.replace('INTARNA_', 'NT_')
            map_name[col] = col_clean
    df_intarna = df_intarna.rename(map_name, axis = 1)


    assert set(df_intarna.columns) - set(other.columns) == set()


    for col in (set(other.columns) - set(df_intarna.columns)):
        df_intarna[col] = np.nan


    df_intarna = pd.DataFrame(df_intarna.mean()).T
    df_intarna['model'] = 'INTARNA'
    df_intarna = df_intarna.filter(other.columns, axis = 1)
    df_full = pd.concat([other, df_intarna], axis = 0).reset_index(drop = True)
    
    
    map_name = {}
    for col in df_full.columns:
        if 'NTconf' in col:
            col_clean = col.replace('NTconf', 'conf')
            map_name[col] = col_clean
        elif 'NT_' in col:
            col_clean = col.replace('NT_', '')
            map_name[col] = col_clean
    df_full = df_full.rename(map_name, axis = 1).reset_index(drop = True)

    return df_full, name_map, confidence_level