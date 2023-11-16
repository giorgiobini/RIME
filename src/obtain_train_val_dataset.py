import pandas as pd
import os
import time
import numpy as np
import argparse
import torch
import datetime
import sys
import random
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, '..')
from util.engine import train_one_epoch
from util.engine import evaluate
from models.nt_classifier import build as build_model 
import util.misc as utils
import json
from dataset.data import (
    RNADataset,
    RNADatasetNT,
    RNADatasetNT500,
    EasyPosAugment,
    InteractionSelectionPolicy,
    SmartNegAugment,
    seed_everything,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, processed_files_dir, original_files_dir, rna_rna_files_dir, metadata_dir, embedding_dir

def obtain_train_dataset(easy_pretraining, train_hq, finetuning, min_n_groups_train, max_n_groups_train, scaling_factor = 5):
    if easy_pretraining:
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_easy.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_easy.csv'))
        subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_training_nt_easy.txt")
    else:
        if train_hq:
            df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_HQ.csv'))
            df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_HQ.csv'))
            subset_train_nt = os.path.join(rna_rna_files_dir, 'gene_pairs_training_nt_HQ.txt')
        else:
            df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt.csv'))
            df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt.csv'))
            if finetuning:
                subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_train_val_fine_tuning_nt.txt")
            else:
                subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_training_nt.txt")

    with open(subset_train_nt, "rb") as fp:  # Unpickling
        list_train = pickle.load(fp)

    vc_train = df_nt[df_nt.couples.isin(list_train)].interacting.value_counts()
    assert vc_train[False]>vc_train[True]
    unbalance_factor = 1 - (vc_train[False] - vc_train[True]) / vc_train[False]

    if easy_pretraining:
        pos_multipliers = {15:0.2, 
                25:0.3,
                50:0.2, 
                100:0.23, 
                10_000_000: 0.07}

        neg_multipliers = {15:0.05, 
                        28:0.15,

                        40:0.08,
                        50:0.05,
                        60:0.1,

                        80:0.03,
                        90:0.03,
                        100:0.05,

                        110:0.05,

                        120:0.1,

                        140:0.05,
                        160:0.03,
                        180:0.03,
                        200:0.03,
                        220:0.02,
                        240:0.01,
                        260:0.01,

                        10_000_000: 0.1}
        
    else:
        pos_multipliers = {15:0.2, 
                       25:0.3,
                       50:0.2, 
                       100:0.23,
                       100_000_000:0.07}
        neg_multipliers = pos_multipliers

    policies_train = [
        EasyPosAugment(
            per_sample=0.5,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=pos_multipliers,
            height_multipliers=pos_multipliers,
        ),  
        SmartNegAugment(
            per_sample=unbalance_factor * 0.5,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=neg_multipliers,
            height_multipliers=neg_multipliers,
        ),
    ] 
        
    dataset_train = RNADatasetNT(
            gene2info=df_genes_nt,
            interactions=df_nt,
            subset_file=subset_train_nt,
            augment_policies=policies_train,
            data_dir = os.path.join(embedding_dir, '32'),
            scaling_factor = scaling_factor,
            min_n_groups = min_n_groups_train,
            max_n_groups = max_n_groups_train,
    )
    
    return dataset_train, policies_train

def obtain_val_dataset(easy_pretraining, finetuning, min_n_groups_val, max_n_groups_val, scaling_factor = 5):

    if easy_pretraining:
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_easy.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_easy.csv'))
        subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_val_sampled_nt_easy.txt")
    else:
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_HQ.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_HQ.csv'))
        if finetuning:
            subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_test_sampled_nt_HQ.txt") # gene_pairs_test_sampled_nt.txt it is also HQ
            df500 = pd.read_csv(os.path.join(metadata_dir, f'test500.csv'))
        else:
            subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_val_sampled_nt_HQ.txt") # gene_pairs_val_sampled_nt.txt it is also HQ
            df500 = pd.read_csv(os.path.join(metadata_dir, f'val500.csv'))

        with open(subset_val_nt, "rb") as fp:  # Unpickling
            list_val = pickle.load(fp)

        assert df500.shape[0] == df_nt[['couples', 'interacting', 'policy']].merge(df500, on = 'couples').shape[0]
        df500 = df_nt[['couples', 'interacting', 'policy']].merge(df500, on = 'couples')
        df500 = df500[df500.couples.isin(list_val)] # in questo modo ho quasi bilanciato del tutto, ma per avere un bilanciamento al 100% devo fare undersampling
        df500 = undersample_df(df500) #bilanciamento al 100%.
        
    if easy_pretraining:
        pos_multipliers = {25:0.7, 50:0.2, 100:0.1}
        neg_multipliers = {33:0.3, 45:0.1, 55:0.1, 65:0.1,
                           80:0.05, 90:0.05, 100:0.05,
                           120:0.05, 150:0.02, 160:0.02,
                           170:0.02, 180:0.02, 190:0.02,
                           200:0.02, 210:0.02, 220:0.02}

        policies_val = [
            EasyPosAugment(
                per_sample=1,
                interaction_selection=InteractionSelectionPolicy.LARGEST,
                width_multipliers=pos_multipliers,
                height_multipliers=pos_multipliers,
            ),  
            SmartNegAugment(
                per_sample=1, # unbalance_factor
                interaction_selection=InteractionSelectionPolicy.LARGEST,
                width_multipliers=neg_multipliers,
                height_multipliers=neg_multipliers,
            ),
        ]

        dataset_val = RNADatasetNT(
            gene2info=df_genes_nt,
            interactions=df_nt,
            subset_file=subset_val_nt,
            augment_policies=policies_val,
            data_dir = os.path.join(embedding_dir, '32'),
            scaling_factor = scaling_factor,
            min_n_groups = min_n_groups_val,
            max_n_groups = max_n_groups_val,
        )

    else:

        df500 = df500.sample(frac=1, random_state=23).reset_index(drop = True)
        assert df500.shape[0]>0

        dataset_val = RNADatasetNT500(
            df = df500,
            data_dir = os.path.join(embedding_dir, '32'),
            scaling_factor = scaling_factor,
            min_n_groups = min_n_groups_val,
            max_n_groups = max_n_groups_val,
        )
        policies_val = 'dataset500'

    return dataset_val, policies_val