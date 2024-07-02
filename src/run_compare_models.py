import pandas as pd
import os
import time
import numpy as np
import seaborn as sns
import pickle
import torch
from pathlib import Path
import argparse
import math
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc
from tqdm.notebook import tqdm
sys.path.insert(0, '..')


from util.evaluation import load_results, map_signal_to_sigmoid_range, balance_df
from util.xai import gradcam, interpolate_expl_matrix, plot_matrix
from config import *
from models.nt_classifier import build as build_model 

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
    
    space = 'linear'
    n_values = 12
    MIN_PERC = 1

    assert space in ['log', 'linear']    
    
    for i in range(1, 1_000):
        df_full, name_map, confidence_level = load_results(checkpoint_dir_paths[:i], space, n_values, MIN_PERC, chkpt_folder)
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python run_compare_models.py &> run_compare_models.out &

    main()