import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
import math
import torch
from tqdm.notebook import tqdm
import seaborn as sns
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.data import (
    RNADataset,
    RNADatasetNT,
    EasyPosAugment,
    InteractionSelectionPolicy,
    EasyNegAugment,
    HardPosAugment,
    HardNegAugment,
    SmartNegAugment,
    plot_sample,
    #plot_sample2,
    seed_everything,
    clean_nt_dataframes_before_class_input,
    get_features1,
    get_features2, 
    get_full_overlap_features1,
    get_full_overlap_features2,
)


def get_df_small_medium_big(df, limit_small_rna = 7_000, limit_medium_rna = 8_500, plot_distributions= False):
    print('limit of small rna is:', limit_small_rna)
    print('limit of medium rna is:', limit_medium_rna)


    small1 = df[(df.length_2<=limit_small_rna)&(df.length_1<=limit_medium_rna)] #both small and one small the other medium
    small2 = df[(df.length_1<=limit_small_rna)&(df.length_2<=limit_medium_rna)] #both small and one small the other medium
    df_small = pd.concat([small1, small2], axis = 0).drop_duplicates()

    df_big = df[(df.length_1>limit_medium_rna)&(df.length_2>limit_medium_rna)]

    big_small_idx = list(df_small.index) + list(df_big.index) 
    df_medium = df.loc[set(df.index) - set(big_small_idx)]

    assert (df_small.shape[0]+df_medium.shape[0]+df_big.shape[0]) == df.shape[0]
    assert set(df_big.couples).union(df_medium.couples).union(df_small.couples) == set(df.couples)

    df_small = df_small.reset_index(drop=True)
    df_medium = df_medium.reset_index(drop=True)
    df_big = df_big.reset_index(drop=True)

    set(df_small.couples).union(set(df_medium.couples)).union(set(df_big.couples)) == set(df.couples)
    
    if plot_distributions:
        # Plotting the KDE plots
        sns.kdeplot(np.array(pd.concat([df_small['length_1'], df_small['length_2']], axis=0)), color='red', label='Small RNAs dataframe')
        sns.kdeplot(np.array(pd.concat([df_medium['length_1'], df_medium['length_2']], axis=0)), color='blue', label='Medium RNAs dataframe')
        sns.kdeplot(np.array(pd.concat([df_big['length_1'], df_big['length_2']], axis=0)), color='green', label='Large RNAs dataframe')

        # Improving the title and labels
        plt.title('Length Distribution of Small, Medium, and Large RNAs')
        plt.xlabel('RNA Length')
        plt.ylabel('Density')

        # Limit the x-axis
        plt.xlim(0, 20000)

        # Display the legend
        plt.legend()

        # Show the plot
        plt.show()
    
    return df_small, df_medium, df_big

def calc_perc_small_medium_big(df_small_shape, df_medium_shape, df_big_shape, df_shape):
    perc_small = np.round(df_small_shape/df_shape*100, 1)
    perc_medium = np.round(df_medium_shape/df_shape*100, 1)
    perc_big = np.round(df_big_shape/df_shape*100, 1)
    
    print(f'The amount of contact matrixes (in the entire dataset) that are small is {perc_small}% ')
    print(f'The amount of contact matrixes (in the entire dataset) that are medium is {perc_medium}% ')
    print(f'The amount of contact matrixes (in the entire dataset) that are big is {perc_big}% ')
    return perc_small, perc_medium, perc_big

def create_sample_dict(id_couple, sample):
    """
    x1, x2, y1, y2 is where is the interaction with respect to the actual coordinates (not the original coordinates)
    length_1, length_2 is the length with respect to the actual length (not the original length)
    
    original_x1, original_x2, original_y1, original_y2 is where the rna interacts with respect to original coordinates 
    window_x1, window_x2, window_y1, window_y2 is where the rna was sampled with respect to original coordinates 
    original_length1, original_length2 is the original length
    """
    d = {
        'id_sample':id_couple,
        'couples':sample.couple_id,
        'gene1':sample.gene1,
        'gene2':sample.gene2,
        'interacting':sample.interacting,
        'length_1':sample.bbox.x2-sample.bbox.x1,
        'length_2':sample.bbox.y2-sample.bbox.y1,
        'protein_coding_1':sample.gene1_info["protein_coding"],
        'protein_coding_2':sample.gene2_info["protein_coding"],
        'x1': sample.seed_interaction_bbox.x1 - sample.bbox.x1,
        'x2': sample.seed_interaction_bbox.x2 - sample.bbox.x1,
        'y1': sample.seed_interaction_bbox.y1 - sample.bbox.y1,
        'y2': sample.seed_interaction_bbox.y2 - sample.bbox.y1,
        'policy':sample.policy,
        'cdna1':sample.gene1_info["cdna"][sample.bbox.x1:sample.bbox.x2],
        'cdna2':sample.gene2_info["cdna"][sample.bbox.y1:sample.bbox.y2],
        'original_x1':sample.seed_interaction_bbox.x1,
        'original_x2':sample.seed_interaction_bbox.x2,
        'original_y1':sample.seed_interaction_bbox.y1,
        'original_y2':sample.seed_interaction_bbox.y2,
        'window_x1':sample.bbox.x1,
        'window_x2':sample.bbox.x2,
        'window_y1':sample.bbox.y1,
        'window_y2':sample.bbox.y2,
        'original_length1':len(sample.gene1_info["cdna"]),
        'original_length2':len(sample.gene2_info["cdna"]),
        'id_gene1_sample':sample.gene1 + '_' + str(sample.bbox.x1) + '_' + str(sample.bbox.x2),
        'id_gene2_sample':sample.gene2 + '_' + str(sample.bbox.y1) + '_' + str(sample.bbox.y2)
    }
    return d

def get_dataset(ep_per_sample, sn_per_sample, en_persample, hn_per_sample, df_genes, df, subset_file, pos_width_multipliers, pos_height_multipliers, neg_width_windows, neg_height_windows):
    assert np.round(sum(pos_width_multipliers.values()), 4) == np.round(sum(neg_width_windows.values()), 4) == 1
    
    pol = []
    
    if ep_per_sample>0:
        pol.append(
            EasyPosAugment(
            per_sample=ep_per_sample,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=pos_width_multipliers,
            height_multipliers=pos_height_multipliers,
            )
        )
        
    if sn_per_sample>0:
        pol.append(   
            SmartNegAugment(
                per_sample=sn_per_sample,
                interaction_selection=InteractionSelectionPolicy.LARGEST,
                width_multipliers=pos_width_multipliers,
                height_multipliers=pos_height_multipliers,
            )
        )
        
    if en_persample>0:
        pol.append( 
            EasyNegAugment(
                per_sample=en_persample,
                width_windows=neg_width_windows,
                height_windows=neg_height_windows,
            )
        )
        
    if hn_per_sample>0:
        pol.append(
            HardNegAugment(
                per_sample=hn_per_sample,
                width_windows=neg_width_windows,
                height_windows=neg_height_windows,
            )
        )
        
    dataset = RNADataset(
            gene2info=df_genes,
            interactions=df,
            subset_file=subset_file,
            augment_policies=pol,
    )
    return dataset 