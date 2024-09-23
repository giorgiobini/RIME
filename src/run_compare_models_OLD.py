import pandas as pd
import os
import time
import numpy as np
import seaborn as sns
import pickle
import torch
from tqdm.notebook import tqdm
from pathlib import Path
import argparse
import math
from scipy import stats
import copy
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, '..')

from util.evaluation import load_paris_results, load_ricseq_splash_mario_results, map_signal_to_sigmoid_range, balance_df, plot_histogram_01, remove_outliers, log_func, load_res_and_tools, plot_all_model_auc, plot_all_model_auc, obtain_all_model_auc, obtain_all_model_auc_patches, obtain_subset_from_task, plot_results_of_all_models, obtain_sr_nosr, make_plot_kde_and_test_difference, calculate_correlations
from util.plot_utils import plot_results_based_on_treshold_for_all_models, plot_results_based_on_topbottom_for_all_models, plot_results_how_many_repeats_in_pred_pos_for_all_models, plot_metric_confidence_for_all_models, plot_tnr_based_on_distance_for_all_models, plot_confidence_based_on_distance_for_all_models, plot_tnr_for_all_models, quantile_bins, plot_features_vs_risearch2_confidence, plot_heatmap
from config import *
from models.nt_classifier import build as build_model 
from util.evaluation import *

external_dataset_dir = os.path.join(dataset_files_dir, 'external_dataset', '500_test_tables')


def load_res_and_tools(external_dataset_dir, checkpoint_dir, tools, dataset, how, only_test, exclude_train_genes, exclude_paris_genes, exclude_paris_couples, filter_hq_ricseq, MIN_N_READS_RICSEQ, specie_paris, paris_hq, paris_hq_threshold):
    
    if type(checkpoint_dir) == str:
        checkpoint_dir = [checkpoint_dir]
    else:
        assert type(checkpoint_dir) == list
        assert type(checkpoint_dir[0]) == str

    if dataset == 'paris':
        
        test500 = pd.read_csv(os.path.join(metadata_dir, f'test500.csv'))
        test500['distance_from_site'] = ( (test500['distance_x'] ** 2) + (test500['distance_y']** 2) )**(0.5) #pitagora
        test500['distance_from_site_embedding'] = ( (test500['distance_embedding_x'] ** 2) + (test500['distance_embedding_y']** 2) )**(0.5) #pitagora
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_HQ.csv'))
        assert test500.shape[0] == df_nt[['couples', 'interacting']].merge(test500, on = 'couples').shape[0]
        test500 = df_nt[['couples', 'interacting']].merge(test500, on = 'couples').reset_index(drop = True)
        
        for i in range(len(checkpoint_dir)):
            r = load_paris_results(checkpoint_dir[i], test500, df_nt, 'test', specie_paris)
            if paris_hq:
                r = filter_hq_data_by_interaction_length(r, test500, paris_hq_threshold)
                
            if i == 0:
                res = r.copy()
            else:
                r = r.rename({'probability':f'nt{i}'}, axis = 1)
                res = pd.concat([res, r[f'nt{i}']], axis = 1)
    else:
        test500 = pd.read_csv(os.path.join(metadata_dir, f'{how}500.csv'))
        test500['distance_from_site'] = ( (test500['distance_x'] ** 2) + (test500['distance_y']** 2) )**(0.5) #pitagora
        test500['distance_from_site_embedding'] = ( (test500['distance_embedding_x'] ** 2) + (test500['distance_embedding_y']** 2) )**(0.5) #pitagora
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_{how}.csv'))
        assert df_nt[['couples', 'interacting']].drop_duplicates().shape[0] == df_nt.shape[0]
        test500 = df_nt[['couples', 'interacting']].merge(test500, on = 'couples') #questo interacting verra poi
        
        for i in range(len(checkpoint_dir)):
            r = load_ricseq_splash_mario_results(checkpoint_dir[i], test500, df_nt, how, only_test, exclude_train_genes, exclude_paris_genes, exclude_paris_couples, filter_hq_ricseq, MIN_N_READS_RICSEQ)
            if i == 0:
                res = r.copy()
            else:
                r = r.rename({'probability':f'nt{i}'}, axis = 1)
                res = pd.concat([res, r[f'nt{i}']], axis = 1)
    
    assert (res['ground_truth'] == res['interacting'].astype(int)).all()
    
    
    for tool_name in tools:
        tool = pd.read_csv(os.path.join(external_dataset_dir, f'{tool_name}_{how}500.csv'), sep = ',').fillna(0)
        tool['value'] = tool['value'].astype(float)
        assert (tool.minimum == True).all()
        res = res.merge(tool[['value', 'couples']].rename({'couples':'id_sample', 'value':tool_name}, axis =1), on = 'id_sample', how = 'left').fillna(0)

    
    #mi serve solo per il merge, poi le elimino queste colonne
    if dataset == 'splash':
        df_nt['simple_repeats'] = np.nan
        df_nt['sine_alu'] = np.nan 
        df_nt['low_complex'] = np.nan 
        df_nt['n_reads'] = np.nan 
    
    #now we merge with ENHN500
    testenhn500 = pd.read_csv(os.path.join(metadata_dir, f'{how}ENHN500.csv'))
    testenhn500['distance_from_site'] = ( (testenhn500['distance_x'] ** 2) + (testenhn500['distance_y']** 2) )**(0.5) #pitagora
    testenhn500['distance_from_site_embedding'] = ( (testenhn500['distance_embedding_x'] ** 2) + (testenhn500['distance_embedding_y']** 2) )**(0.5) #pitagora
    testenhn500 = df_nt[['couples', 'where', 'where_x1', 'where_y1', 'simple_repeats', 'sine_alu', 'low_complex']].merge(testenhn500, on = 'couples')
    
    for i in range(len(checkpoint_dir)):
        r = pd.read_csv(os.path.join(checkpoint_dir[i], f'{how}ENHN_results500.csv')).drop('policy', axis = 1)
        # qui e necessario perche 'ground_truth' è preso da df_nt, che originariamente mette positivi gli hardneg. 
        # invece in ENHN500 quello che in df_nt era un positivo diventa un negativo (hardneg, easyneg). comunque queste colonne verranno sovrascritte dopo
        r['ground_truth'] = 0
        r['interacting'] = False
        if i == 0:
            enhn = r.copy()
        else:
            r = r.rename({'probability':f'nt{i}'}, axis = 1)
            enhn = pd.concat([enhn, r[f'nt{i}']], axis = 1)
    
    if dataset == 'ricseq':
        #aggiungi n_reads a testenhn500, quindi a enhn (visto che poi lo mergio con testenhn500)
        to_merge = res[['id_sample', 'n_reads']].rename({'id_sample':'couples'}, axis = 1).drop_duplicates().reset_index(drop=True)
        assert len(set(to_merge.couples)) == to_merge.shape[0]
        testenhn500 = testenhn500.merge(to_merge)
                
           
    #tengo solo le couples di res, perche escludo quelle che erano tipo nel training di paris, oppure solo paris human, oppure quelle < n_reads_ricseq, ecc... 
    enhn = enhn[enhn.couples.isin(res.couples)].reset_index(drop = True)
    
    #policy tengo quello di testenhn500, che sono hn, en
    enhn = enhn.merge(testenhn500.drop(['g1', 'g2'], axis = 1).rename({'couples':'id_sample'}, axis = 1), on = 'id_sample').reset_index(drop = True)

    enhnintarna = load_intarnaENHN500(how)
    enhn = enhn.merge(enhnintarna[['E','E_norm', 'couples']].rename({'couples':'id_sample'}, axis =1), on = 'id_sample', how = 'left').fillna(0)
    enhn['original_area'] = enhn.original_length1 * enhn.original_length2    

    how = how + 'ENHN'
    for tool_name in tools:
        tool = pd.read_csv(os.path.join(external_dataset_dir, f'{tool_name}_{how}500.csv'), sep = ',').fillna(0)
        tool['value'] = tool['value'].astype(float)
        assert (tool.minimum == True).all()
        enhn = enhn.merge(tool[['value', 'couples']].rename({'couples':'id_sample', 'value':tool_name}, axis =1), on = 'id_sample', how = 'left')

    # tolgo 'E_norm_conf', 'ensemble_score'
    res = res.filter(list(enhn.columns), axis = 1)
    
    res = pd.concat([res, enhn], axis = 0).reset_index(drop=True)
    
    res = res.drop(['Unnamed: 0', 'simple_repeats', 'sine_alu', 'low_complex'], axis = 1)
    res['dataset'] = dataset
    
    
    # l unica cosa affidabile è la colonna policy, che viene da test500, testenhn500
    res.loc[res.policy.isin(['easyneg', 'hardneg', 'smartneg']), 'interacting'] = False
    res.loc[res.policy.isin(['easyneg', 'hardneg', 'smartneg']), 'ground_truth'] = 0
    res.loc[res.policy.isin(['easypos']), 'interacting'] = True
    res.loc[res.policy.isin(['easypos']), 'ground_truth'] = 1
    
    assert res[res.ground_truth.isna()].shape[0] == res[res.interacting.isna()].shape[0] == 0
    assert (res['ground_truth'] == res['interacting'].astype(int)).all()
    
    return res


def load_test_set_df(external_dataset_dir, checkpoint_dir, tools, dataset, args_datasets):
    
    assert dataset in ['psoralen', 'paris', 'splash', 'mario', 'ricseq']

    if (dataset == 'psoralen'):
        
        dataset, how = 'paris',  'test'
        paris = load_res_and_tools(external_dataset_dir, checkpoint_dir, tools, dataset, how, 
                                   args_datasets[how]['only_test'], 
                                   args_datasets[how]['exclude_train_genes'], 
                                   args_datasets[how]['exclude_paris_genes'], 
                                   args_datasets[how]['exclude_paris_couples'],
                                   args_datasets[how]['filter_hq_ricseq'], 
                                   args_datasets[how]['MIN_N_READS_RICSEQ'], 
                                   args_datasets[how]['SPECIE_PARIS'],
                                   args_datasets[how]['PARIS_HQ'],
                                   args_datasets[how]['PARIS_HQ_THRESHOLD'],)
        
        dataset, how = 'splash', 'splash'
        splash = load_res_and_tools(external_dataset_dir, checkpoint_dir, tools, dataset, how, 
                                   args_datasets[how]['only_test'], 
                                   args_datasets[how]['exclude_train_genes'], 
                                   args_datasets[how]['exclude_paris_genes'], 
                                   args_datasets[how]['exclude_paris_couples'],
                                   args_datasets[how]['filter_hq_ricseq'], 
                                   args_datasets[how]['MIN_N_READS_RICSEQ'], 
                                   args_datasets[how]['SPECIE_PARIS'],
                                   args_datasets[how]['PARIS_HQ'],
                                   args_datasets[how]['PARIS_HQ_THRESHOLD'],)
                  
        assert set(paris.columns) == set(splash.columns)
        res = pd.concat([paris, splash], axis = 0).reset_index(drop=True)
    
    else: 
        how = dataset if dataset != 'paris' else 'test'
        res = load_res_and_tools(external_dataset_dir, checkpoint_dir, tools, dataset, how, 
                                   args_datasets[how]['only_test'], 
                                   args_datasets[how]['exclude_train_genes'], 
                                   args_datasets[how]['exclude_paris_genes'], 
                                   args_datasets[how]['exclude_paris_couples'],
                                   args_datasets[how]['filter_hq_ricseq'], 
                                   args_datasets[how]['MIN_N_READS_RICSEQ'], 
                                   args_datasets[how]['SPECIE_PARIS'],
                                   args_datasets[how]['PARIS_HQ'],
                                   args_datasets[how]['PARIS_HQ_THRESHOLD'],)
                  
    return res


def plot_sr_distributions(df_sr, label_x, figsize = (16, 8)):

    # Create the violin plot without the inner box plot
    plt.figure(figsize=figsize)
    ax = sns.violinplot(x='Model', y='Normalized Score', hue='Category', data=df_sr, split=True, palette=['#FF9999', '#99FF99'], inner=None)

    # Add the mean points with custom horizontal lines
    mean_line_length = 0.3  # Adjust this value to control the length of the horizontal lines

    # Calculate means
    mean_points = df_sr.groupby(['Model', 'Category'])['Normalized Score'].mean().reset_index()

    # Get the positions of each category for plotting
    positions = {category: idx for idx, category in enumerate(df_sr['Model'].unique())}

    # Plot mean lines manually
    for i, model in enumerate(mean_points['Model'].unique()):
        for j, category in enumerate(mean_points['Category'].unique()):
            mean_val = mean_points[(mean_points['Model'] == model) & (mean_points['Category'] == category)]['Normalized Score'].values[0]
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

    plt.title('Violin Plot with Two Distributions per Category')
    plt.show()
    
def obtain_epsn_ephnen_from_res(res):
    easypos_smartneg = res[(res.policy.isin(['easypos', 'smartneg']))].reset_index(drop = True)
    ephnen = res[
        ((res.distance_from_site_embedding > 0)&(res.policy.isin(['hardneg', 'easyneg']))) |
        (res.policy == 'easypos')
    ].reset_index(drop = True)
    enhn500 = res[(res.distance_from_site_embedding == 0)&(res.policy.isin(['easypos', 'hardneg', 'easyneg']))].reset_index(drop = True)
    return easypos_smartneg, ephnen, enhn500



def otain_results_new(checkpoint_dir_paths, index_name):
    diz_results = {}
    name_map = {}
    
    for idx_v, checkpoint_dir in tqdm(enumerate(checkpoint_dir_paths), total = len(checkpoint_dir_paths)):
        row = {}
        
        datasets = ['paris', 'ricseq', 'splash', 'mario']
        tools = [] #['priblast', 'RNAplex', 'rnacofold', 'assa', 'risearch2', 'RNAhybrid', 'RNAup'] #['priblast', 'risearch2'] #['priblast', 'RNAup', 'RNAplex', 'RNAhybrid', 'rnacofold', 'risearch2', 'assa']
        energy_columns = []

        #ARGS for paris
        SPECIE_PARIS = 'human'
        PARIS_HQ = False
        PARIS_HQ_THRESHOLD = 35
        MIN_N_READS_RICSEQ = 2
        RICSEQ_HQ = False

        RICSEQ_TEST = False
        SPLASH_TEST = True

        #ARGS for splash, ricseq, mario


        args_datasets = {'paris':
                         {'only_test' : np.nan, #uneuseful for paris
                          'exclude_train_genes' : np.nan, #uneuseful for paris
                          'exclude_paris_genes' : np.nan, #uneuseful for paris
                          'exclude_paris_couples' : np.nan, #uneuseful for paris
                          'filter_hq_ricseq' : np.nan, #uneuseful for paris
                          'MIN_N_READS_RICSEQ' : np.nan, #uneuseful for paris
                          'SPECIE_PARIS' : SPECIE_PARIS,
                          'PARIS_HQ':PARIS_HQ,
                          'PARIS_HQ_THRESHOLD':PARIS_HQ_THRESHOLD,
                         }, 
                         'ricseq': 
                         {'only_test' : RICSEQ_TEST,
                          'exclude_train_genes' : False,
                          'exclude_paris_genes' : False,
                          'exclude_paris_couples' : True,
                          'filter_hq_ricseq' : RICSEQ_HQ,
                          'MIN_N_READS_RICSEQ' : MIN_N_READS_RICSEQ,
                          'SPECIE_PARIS' : np.nan, #uneuseful for ricseq
                          'PARIS_HQ':np.nan, #uneuseful for ricseq
                          'PARIS_HQ_THRESHOLD':np.nan, #uneuseful for ricseq
                         },
                         'splash': 
                         {'only_test' : SPLASH_TEST,
                          'exclude_train_genes' : False,
                          'exclude_paris_genes' : False,
                          'exclude_paris_couples' : True,
                          'filter_hq_ricseq' : False, #uneuseful for splash
                          'MIN_N_READS_RICSEQ' : np.nan, #uneuseful for splash
                          'SPECIE_PARIS' : np.nan, #uneuseful for splash
                          'PARIS_HQ':np.nan, #uneuseful for splash
                          'PARIS_HQ_THRESHOLD':np.nan, #uneuseful for splash
                         },
                        'mario': 
                         {'only_test' : False,
                          'exclude_train_genes' : False,
                          'exclude_paris_genes' : False,
                          'exclude_paris_couples' : True,
                          'filter_hq_ricseq' : False, #uneuseful for mario
                          'MIN_N_READS_RICSEQ' : np.nan,  #uneuseful for mario
                          'SPECIE_PARIS' : np.nan, #uneuseful for mario
                          'PARIS_HQ':np.nan, #uneuseful for mario
                          'PARIS_HQ_THRESHOLD':np.nan, #uneuseful for mario
                         },
        }

        args_datasets['test'] = args_datasets['paris'] #sometimes I want to call it 'test'
    
        dfs = []
        for dataset in ['psoralen', 'ricseq', 'mario']:

            res = load_test_set_df(external_dataset_dir, checkpoint_dir, tools, dataset, args_datasets)

            easypos_smartneg, ephnen, enhn500 = obtain_epsn_ephnen_from_res(res) 

            dfs.append(obtain_all_model_auc(easypos_smartneg, tools).rename({'auc': f'auc_EPSN_{dataset}'}, axis = 1))
            dfs.append(obtain_all_model_auc_patches(res, tools).rename({'auc': f'auc_EPENHN_{dataset}'}, axis = 1))

        df_auc = pd.concat(dfs, axis = 1)
        df_auc = df_auc.loc[:,~df_auc.columns.duplicated()].copy()
        row.update(dict(df_auc[df_auc['model_name'] == 'NT'].drop('model_name', axis =1).iloc[0]))

        list_of_models_to_test = ['nt'] + energy_columns

        dfs = []

        for dataset in ['psoralen', 'ricseq', 'mario']:
            res = load_test_set_df(external_dataset_dir, checkpoint_dir, tools, dataset, args_datasets)

            easypos_smartneg, ephnen, enhn500 = obtain_epsn_ephnen_from_res(res) 


            print(f'---------- DATASET: {dataset} ----------')

            subset = ephnen[ephnen.ground_truth == 0].copy().reset_index(drop = True)

            row[f'tnr_enhn_{dataset}'] = (subset.ground_truth == (subset['probability'] > 0.5).astype(int)).sum() / subset.shape[0] 

            subset = enhn500[enhn500.ground_truth == 0].copy().reset_index(drop = True)

            tnr = (subset.ground_truth == (subset['probability'] > 0.5).astype(int)).sum() / subset.shape[0] 

            row[f'tnr_enhn500_{dataset}'] = (subset.ground_truth == (subset['probability'] > 0.5).astype(int)).sum() / subset.shape[0] 

        list_of_models_to_test = ['nt'] + energy_columns
        n_runs_patches = 3
        MIN_PERC = 5
        n_values = 3
        n_run_undersampling = 5

        for dataset in ['psoralen', 'ricseq', 'mario']:

            res = load_test_set_df(external_dataset_dir, checkpoint_dir, tools, dataset, args_datasets)

            easypos_smartneg, ephnen, enhn500 = obtain_epsn_ephnen_from_res(res) 

            #interactors
            print(dataset, 'interactors')
            precision_data, _, model_names = collect_results_based_on_topbottom_for_all_models(easypos_smartneg, MIN_PERC, list_of_models_to_test, n_values, n_run_undersampling, 'precision')
            npv_models, _, model_names = collect_results_based_on_topbottom_for_all_models(easypos_smartneg, MIN_PERC, list_of_models_to_test, n_values, n_run_undersampling, 'npv')

            percs = _.astype(int).astype(str)

            #save precision_data, npv_models
            for i in range(n_values):
                row[f'prec_interactors_{percs[0][i]}_{dataset}'] = np.round(precision_data[0][i], 2)
                row[f'npv_interactors_{percs[0][i]}_{dataset}'] = np.round(npv_models[0][i], 2)


            #patches
            print(dataset, 'patches')

            pos = res[(res.policy == 'easypos')].reset_index(drop=True)
            neg_close = res[((res.distance_from_site_embedding == 0) & (res.policy.isin(['hardneg', 'easyneg'])))].reset_index(drop=True)
            neg_far = res[((res.distance_from_site_embedding > 0) & (res.policy.isin(['hardneg', 'easyneg'])))].reset_index(drop=True)

            prec_list = []
            npv_list = []

            # Find the minimum size among the datasets
            min_size = min(pos.shape[0], neg_close.shape[0], neg_far.shape[0])

            for _ in range(n_runs_patches): 

                # Undersample each dataset to the minimum size
                pos_sample = pos.sample(n=min_size, random_state=np.random.randint(0, 10000))
                neg_close_sample = neg_close.sample(n=min_size, random_state=np.random.randint(0, 10000))
                neg_far_sample = neg_far.sample(n=min_size, random_state=np.random.randint(0, 10000))

                # Combine the undersampled datasets
                balanced_subset = pd.concat([pos_sample, neg_close_sample, neg_far_sample])

                precision_data, _, model_names = collect_results_based_on_topbottom_for_all_models(balanced_subset, MIN_PERC, list_of_models_to_test, n_values, n_run_undersampling, 'precision')
                npv_models, _, model_names = collect_results_based_on_topbottom_for_all_models(balanced_subset, MIN_PERC, list_of_models_to_test, n_values, n_run_undersampling, 'npv')

                prec_list.append(precision_data)
                npv_list.append(npv_models)

            precision_data = np.mean(prec_list, axis = 0)
            npv_models = np.mean(npv_list, axis = 0)

            #save precision_data, npv_models
            for i in range(n_values):
                row[f'prec_patches_{percs[0][i]}_{dataset}'] = np.round(precision_data[0][i], 2)
                row[f'npv_patches_{percs[0][i]}_{dataset}'] = np.round(npv_models[0][i], 2)

            #save precision_data, npv_models

        dfs = []

        n_positives_run = []

        list_of_n_reads_ricseq = [2,3,4,5,6,7]

        for N_READS_RICSEQ in list_of_n_reads_ricseq:

            args_copy = copy.deepcopy(args_datasets)
            args_copy['ricseq']['MIN_N_READS_RICSEQ'] = N_READS_RICSEQ

            res = load_test_set_df(external_dataset_dir, checkpoint_dir, tools, 'ricseq', args_copy)

            easypos_smartneg, ephnen, enhn500 = obtain_epsn_ephnen_from_res(res) 

            pos = easypos_smartneg[easypos_smartneg.policy == 'easypos'].reset_index(drop = True)
            n_positives_run.append(pos.shape[0])

            dfs.append(obtain_all_model_auc(easypos_smartneg, tools).rename({'auc': f'auc_interactors_ricseq_nread{N_READS_RICSEQ}'}, axis = 1))
            dfs.append(obtain_all_model_auc_patches(res, tools).rename({'auc': f'auc_patches_ricseq_nread{N_READS_RICSEQ}'}, axis = 1))

        df_auc = pd.concat(dfs, axis = 1)
        df_auc = df_auc.loc[:,~df_auc.columns.duplicated()].copy()

        row.update(dict(df_auc[df_auc['model_name'] == 'NT'].drop('model_name', axis =1).iloc[0]))
    
        model_name = f'model{idx_v + index_name}'
        name_map[model_name] = checkpoint_dir
        diz_results[model_name] = row
        
    df = pd.DataFrame.from_dict(diz_results, 'index')
    df = df*100
    df = df.round(2)
    df = df.reset_index().rename({'index':'model'}, axis = 1)
    
    return df, name_map

def load_results_new(checkpoint_dir_paths, chkpt_folder):
    
    filename_output = os.path.join(chkpt_folder, 'model_performance_new.pkl')
    
    if os.path.exists(filename_output):
        # Loading the variables from the file
        with open(filename_output, 'rb') as file:
            df, name_map = pickle.load(file)
        
        to_do = list( set(checkpoint_dir_paths) - set(name_map.values()) )
        
        if len(to_do) > 0:
        
            new_df, new_name_map = otain_results_new(to_do, index_name = len(name_map))

            df = pd.concat([df, new_df], axis = 0).reset_index(drop = True)
            name_map.update(new_name_map)

            # Saving the variables to a file
            with open(filename_output, 'wb') as file:
                pickle.dump((df, name_map), file)

    else:
        df, name_map = otain_results_new(checkpoint_dir_paths, index_name = 0)
        # Saving the variables to a file
        with open(filename_output, 'wb') as file:
            pickle.dump((df, name_map), file)
        
    return df, name_map

def main():

    checkpoint_dir_paths = []

    chkpt_folder = os.path.join(ROOT_DIR, 'checkpoints')

    models_to_check = os.listdir(chkpt_folder)
    for model_name in models_to_check:
        model_folder = os.path.join(chkpt_folder, model_name)
        test_paris = os.path.join(chkpt_folder, model_name, 'test_results500.csv')
        ricseq = os.path.join(chkpt_folder, model_name, 'ricseq_results500.csv')
        splash = os.path.join(chkpt_folder, model_name, 'splash_results500.csv')

        testENHN_paris = os.path.join(chkpt_folder, model_name, 'testENHN_results500.csv')
        ricseqENHN = os.path.join(chkpt_folder, model_name, 'ricseqENHN_results500.csv')
        splashENHN = os.path.join(chkpt_folder, model_name, 'splashENHN_results500.csv')
        if os.path.exists(test_paris) & os.path.exists(ricseq) & os.path.exists(splash) & os.path.exists(testENHN_paris) & os.path.exists(ricseqENHN) & os.path.exists(splashENHN) :
            checkpoint_dir_paths.append(model_folder)
    
    for i in range(1, 1_000):
        df, name_map = load_results_new(checkpoint_dir_paths[:i], chkpt_folder)
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python run_compare_models.py &> run_compare_models.out &

    main()