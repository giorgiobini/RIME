import pandas as pd
import os
import time
import numpy as np
import seaborn as sns
import pickle
import torch
import re
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
from mpl_toolkits.axes_grid1 import make_axes_locatable


sys.path.insert(0, '..')

from util.evaluation import ModelResultsManager, calculate_correlations, balancing_only_for_one_task, obtain_all_model_auc, remove_outliers, map_thermodynamic_columns, obtain_df_auc, replace_outliers_with_nan_and_make_positive, obtain_sr_nosr, map_dataset_to_hp
from util.plot_utils import npv_precision, plot_sr_distributions, calc_metric, plot_tnr_based_on_distance_for_our_model, collect_results_based_on_topbottom_for_all_models, plot_results_based_on_topbottom_for_all_models, plot_results_how_many_repeats_in_pred_pos_for_all_models, plot_metric_confidence_for_all_models, plot_tnr_based_on_distance_for_all_models, plot_confidence_based_on_distance_for_all_models, plot_tnr_for_all_models, quantile_bins, plot_features_vs_risearch2_confidence, plot_heatmap, plot_tnr_recall_for_all_models, plot_correlation_nreads_prob_intsize
from util.model_names_map import map_model_names
from config import *
from models.nt_classifier import build as build_model 

external_dataset_dir = os.path.join(dataset_files_dir, 'external_dataset', '200_test_tables')

def otain_results_new(checkpoint_dir_paths, index_name, n_run_undersampling = 15):
    
    PARIS_FINETUNED_MODEL = True
    SPLASH_TRAINED_MODEL = True
    energy_columns = [] #['IntaRNA', 'priblast', 'RNAplex', 'rnacofold', 'assa', 'RNAhybrid', 'RNAup', 'risearch2']
    list_of_datasets = ['val', 'val_HQ', 'val_mouse', 'psoralen_val', 'parisHQ', 'paris_mouse_HQ', 'ricseqHQ', 'psoralen', 'paris', 'paris_mouse', 'ricseq', 'mario', 'splash']
    
    diz_results = {}
    name_map = {}
    
    for idx_v, checkpoint_dir in tqdm(enumerate(checkpoint_dir_paths), total = len(checkpoint_dir_paths)):
        row = {}
        
        model_name = re.search(r'[^/]+$', checkpoint_dir).group(0)
        
        modelRM = ModelResultsManager(
            model_name = model_name,
            dimension = 200, 
            chkpt_directory = os.path.join(ROOT_DIR, 'checkpoints'), 
            rna_rna_files_dir = rna_rna_files_dir, 
            test_info_directory = metadata_dir, 
            other_tools = energy_columns, 
            other_tools_dir = external_dataset_dir
        )

        df_auc = obtain_df_auc(modelRM, PARIS_FINETUNED_MODEL, energy_columns, SPLASH_TRAINED_MODEL, list_of_datasets = list_of_datasets, logistic_regression_models = {} )

        for dataset in list_of_datasets:
            
            row[f'auc_interactors_{dataset}'] = df_auc[df_auc['model_name'] == MODEL_NAME][f'auc_interactors_{dataset}'].iloc[0]
            row[f'auc_patches_{dataset}'] = df_auc[df_auc['model_name'] == MODEL_NAME][f'auc_patches_{dataset}'].iloc[0]
            
            experiment, specie_paris, paris_hq_threshold, n_reads_ricseq, n_reads_paris, interlen_OR_nreads_paris, paris_test = map_dataset_to_hp(dataset)
    
            if dataset in ['parisHQ', 'val_HQ', 'paris_mouse_HQ', 'paris_mouse_HQ', 'ricseqHQ', 'mario', 'splash']:
                n_run_undersampling = 50 
            else:
                n_run_undersampling = 10

            res = modelRM.get_experiment_data(
                experiment = experiment, 
                paris_test = paris_test, 
                paris_finetuned_model = PARIS_FINETUNED_MODEL, 
                specie_paris = specie_paris,
                paris_hq = False,
                paris_hq_threshold = paris_hq_threshold,
                n_reads_paris = n_reads_paris,
                interlen_OR_nreads_paris = interlen_OR_nreads_paris,
                splash_trained_model = SPLASH_TRAINED_MODEL,
                only_test_splash_ricseq_mario = False,
                n_reads_ricseq = n_reads_ricseq,
                logistic_regression_models = {},
            )
            
            npv_list = []
            prec_list = []
            spec_list = []
            rec_list = []
            ce_list = []
            ce2_list = []
            ce3_list = []

            for i in range(n_run_undersampling):
                
                balanced = balancing_only_for_one_task(res, task = 'all')
                
                npv_list.append(calc_metric(balanced, 'probability', 'npv'))
                prec_list.append(calc_metric(balanced, 'probability', 'precision'))
                spec_list.append(calc_metric(balanced, 'probability', 'specificity'))
                rec_list.append(calc_metric(balanced, 'probability', 'recall'))      
                ce_list.append(calc_metric(balanced, 'probability', 'cross_entropy'))  
                ce2_list.append(calc_metric(balanced, 'probability', 'cross_entropy_FP'))  
                ce3_list.append(calc_metric(balanced, 'probability', 'cross_entropy_FN'))  
            
            row[f'npv_{dataset}'] = np.round(np.mean(npv_list), 5)
            row[f'precision_{dataset}'] = np.round(np.mean(prec_list), 5)
            row[f'tnr_{dataset}'] = np.round(np.mean(spec_list), 5)
            row[f'recall_{dataset}'] = np.round(np.mean(rec_list), 5)
            row[f'ce_{dataset}'] = np.round(np.mean(ce_list), 5)
            row[f'ceFP_{dataset}'] = np.round(np.mean(ce2_list), 5)
            row[f'ceFN_{dataset}'] = np.round(np.mean(ce3_list), 5)
        
        model_name = f'model{idx_v + index_name}'
        name_map[model_name] = checkpoint_dir
        diz_results[model_name] = row
        
    df = pd.DataFrame.from_dict(diz_results, 'index')
    df = df*100
    df = df.round(5)
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
        test_paris = os.path.join(chkpt_folder, model_name, 'test_results200.csv')
        val_paris = os.path.join(chkpt_folder, model_name, 'val_results200.csv')
        ricseq = os.path.join(chkpt_folder, model_name, 'ricseq_results200.csv')
        splash = os.path.join(chkpt_folder, model_name, 'splash_results200.csv')
        mario = os.path.join(chkpt_folder, model_name, 'mario_results200.csv')
        
        if os.path.exists(test_paris) & os.path.exists(val_paris) & os.path.exists(ricseq) & os.path.exists(splash) & os.path.exists(mario):
            checkpoint_dir_paths.append(model_folder)

    for i in range(1, 1_000):
        df, name_map = load_results_new(checkpoint_dir_paths[:i], chkpt_folder)
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python run_compare_models.py &> run_compare_models.out &

    main()