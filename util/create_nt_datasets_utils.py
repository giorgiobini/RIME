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

def estimate_time_and_space(n_samples):
    
    print('# sequences:', n_samples)
    
    #TIME
    minutes = 3219*n_samples/(228278)
    hours = minutes/60
    days = hours/24
    print('estimated # hours:', np.round(hours, 2))
    print('estimated # days:', np.round(days, 2))

    mb = 10.2*n_samples
    gb = mb/1000
    tb = gb/1000
    print('estimated terabytes (pessimistic):', np.round(tb, 2))
    mb = 1995*n_samples/(300)
    gb = mb/1000
    tb = gb/1000
    print('estimated terabytes (realistic):', np.round(tb, 2))

def get_directory_size(directory):
    total_size = 0
    n_files = 0
    # Walk through all the files and subdirectories in the directory
    for path, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.stat(file_path).st_size
            n_files+=1

    # Convert the size to a human-readable format
    size_in_bytes = total_size
    size_in_kilobytes = total_size / 1000
    size_in_megabytes = size_in_kilobytes / 1000
    size_in_gigabytes = size_in_megabytes / 1000
    size_in_terabytes = size_in_gigabytes / 1000

    return {
        "n_files":n_files,
        # "bytes": size_in_bytes,
        # "kilobytes": size_in_kilobytes,
        # "megabytes": size_in_megabytes,
        "gigabytes": size_in_gigabytes, 
        "terabytes": size_in_terabytes, 
    }



def swap_if_needed(df):
    df['actual_couples'] = df.gene1 + '_' + df.gene2
    df['need_to_swap'] = df.couples!=df.actual_couples
    where = df.need_to_swap
    df.loc[where, ['gene1', 'gene2']] = (df.loc[where, ['gene2', 'gene1']].values)
    df.loc[where, ['length_1', 'length_2']] = (df.loc[where, ['length_2', 'length_1']].values)
    df.loc[where, ['protein_coding_1', 'protein_coding_2']] = (df.loc[where, ['protein_coding_2', 'protein_coding_1']].values)
    df.loc[where, ['x1', 'y1']] = (df.loc[where, ['y1', 'x1']].values)
    df.loc[where, ['x2', 'y2']] = (df.loc[where, ['y2', 'x2']].values)
    df.loc[where, ['cdna1', 'cdna2']] = (df.loc[where, ['cdna2', 'cdna1']].values)
    df.loc[where, ['original_x1', 'original_y1']] = (df.loc[where, ['original_y1', 'original_x1']].values)
    df.loc[where, ['original_x2', 'original_y2']] = (df.loc[where, ['original_y2', 'original_x2']].values)
    df.loc[where, ['window_x1', 'window_y1']] = (df.loc[where, ['window_y1', 'window_x1']].values)
    df.loc[where, ['window_x2', 'window_y2']] = (df.loc[where, ['window_y2', 'window_x2']].values)
    df.loc[where, ['original_length1', 'original_length2']] = (df.loc[where, ['original_length2', 'original_length1']].values)
    df.loc[where, ['id_gene1_sample', 'id_gene2_sample']] = (df.loc[where, ['id_gene2_sample', 'id_gene1_sample']].values)
    if ('diff1' in df.columns)&('diff2' in df.columns):
        df.loc[where, ['diff1', 'diff2']] = (df.loc[where, ['diff2', 'diff1']].values)
    df['actual_couples'] = df.gene1 + '_' + df.gene2
    assert (df.couples==df.actual_couples).all()
    return df.drop(['need_to_swap', 'actual_couples'], axis = 1)

def create_fake_interaction_region(df, interaction_size=16):
    subset = df[df.policy.isin(['hardneg', 'easyneg'])]
    
    length1_values = subset['length_1'].values - interaction_size
    length2_values = subset['length_2'].values - interaction_size

    # Generate random indices within the length1 range
    x1_indices = np.random.randint(0, length1_values, size=len(subset))
    x2_indices = x1_indices + interaction_size  # Ensure a distance of interaction_size between x1 and x2

    # Generate random indices within the length2 range
    y1_indices = np.random.randint(0, length2_values, size=len(subset))
    y2_indices = y1_indices + interaction_size  # Ensure a distance of interaction_size between y1 and y2
    
    df.loc[df.policy.isin(['hardneg', 'easyneg']), 'x1'] = x1_indices
    df.loc[df.policy.isin(['hardneg', 'easyneg']), 'x2'] = x2_indices
    df.loc[df.policy.isin(['hardneg', 'easyneg']), 'y1'] = y1_indices
    df.loc[df.policy.isin(['hardneg', 'easyneg']), 'y2'] = y2_indices
    return df


def clean_df_nt(df_nt):    
    
    df_nt = df_nt.rename({'gene1':'gene1_id', 'gene2':'gene2_id'}, axis = 1)
    df_nt = df_nt.rename({'id_gene1_sample':'gene1', 'id_gene2_sample':'gene2'}, axis = 1)


    df_nt['w'] = df_nt['x2'] - df_nt['x1']
    df_nt['h'] = df_nt['y2'] - df_nt['y1']
    
    column_order = ['couples','gene1','gene2','interacting',
                    'length_1','length_2','protein_coding_1','protein_coding_2',
                    'x1','y1','w','h', 'policy',
                    'original_x1','original_x2',
                    'original_y1','original_y2',
                    'id_gene1_sample','id_gene2_sample', 'couples_id',
                   ]
    df_nt = df_nt.drop_duplicates(subset = [
        'couples','gene1','gene2','interacting',
        'length_1','length_2','protein_coding_1',
        'protein_coding_2','x1','y1','w','h',
        'policy','original_x1','original_x2',
        'original_y1','original_y2','couples_id'
    ]).reset_index(drop = True)
    
    df_nt = df_nt.filter(column_order, axis = 1)
    
    return df_nt

def create_df_genes_nt(df_full):
    
    df_full = df_full.rename({'gene1':'gene1_id', 'gene2':'gene2_id'}, axis = 1)
    df_full = df_full.rename({'id_gene1_sample':'gene1', 'id_gene2_sample':'gene2'}, axis = 1)
    
    column_order = [
        'gene1','gene2','id_gene1_sample','id_gene2_sample',
        'original_length1','original_length2', 'cdna1', 'cdna2',
        'window_x1','window_x2','window_y1','window_y2', 
        'gene1_id', 'gene2_id', 'protein_coding_1',  'protein_coding_2'
    ]
    
    df_g = df_full.filter(column_order, axis = 1)

    df_g1 = df_g.filter(
        [
        'gene1', 
        'id_gene1_sample', 
        'cdna1', 
        'window_x1',
        'window_x2', 
        'gene1_id',
        'protein_coding_1', 
        'length_1',
        'original_length1', 
        ]
    ).rename(
        {
        'gene1':'gene_id',
        'cdna1':'cdna', 
        'length_1':'length',
        'window_x1':'window_c1',
        'window_x2':'window_c2',
        'gene1_id':'original_gene_id', 
        'protein_coding_1':'protein_coding', 
        'original_length1':'original_length'
        }, 
        axis = 1)
    df_g2 = df_g.filter(
        [
        'gene2', 
        'id_gene2_sample', 
        'cdna2', 
        'window_y1',
        'window_y2', 
        'gene2_id',
        'protein_coding_2', 
        'length_2',
        'original_length2', 
        ]
    ).rename(
        {
        'gene2':'gene_id',
        'cdna2':'cdna', 
        'length_2':'length',
        'window_y1':'window_c1',
        'window_y2':'window_c2',
        'gene2_id':'original_gene_id', 
        'protein_coding_2':'protein_coding', 
        'original_length2':'original_length'
        }, 
        axis = 1)

    df_genes_nt = pd.concat([df_g1, df_g2], axis = 0).drop_duplicates().reset_index(drop = True)

    df_genes_nt['UTR5'] = 0
    df_genes_nt['CDS'] = 0
    df_genes_nt['UTR3'] = 0
    
    return df_genes_nt