import pandas as pd
import os
import time
import numpy as np
import pickle
import argparse
import torch
import matplotlib.pyplot as plt
import sys
import datetime
from pathlib import Path
import seaborn as sns
import time
from torch.utils.data import DataLoader
sys.path.insert(0, '..')
from config import *
import util.misc as utils
import util.xai as xai
from models.nt_classifier import build as build_model
from dataset.data import (
    RNADatasetNT,
    EasyPosAugment,
    InteractionSelectionPolicy,
    SmartNegAugment,
    seed_everything,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, processed_files_dir, original_files_dir, rna_rna_files_dir, metadata_dir, embedding_dir

RUN_FINETUNING = True

def main(args):
    
    start_time = time.time() 

    file_test  = os.path.join(rna_rna_files_dir, "gene_pairs_test_nt.txt")
    #file_val  = os.path.join(rna_rna_files_dir, "gene_pairs_val_nt.txt")
    
    df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt.csv'))
    df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt.csv'))

    pos_multipliers = {10_000_000:1.,}
    neg_multipliers = pos_multipliers

    policies_test = [
        EasyPosAugment(
            per_sample=1,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=pos_multipliers,
            height_multipliers=pos_multipliers,
        ),  
        SmartNegAugment(
            per_sample=1,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=neg_multipliers,
            height_multipliers=neg_multipliers,
        ),
    ]

    dataset_test = RNADatasetNT(
        gene2info=df_genes_nt,
        interactions=df_nt,
        subset_file=file_test,
        augment_policies=policies_test,
        data_dir = os.path.join(embedding_dir, '32'),
        scaling_factor = 5,
        min_n_groups = np.nan,
        max_n_groups = MAX_N_GROUP_SIZE,
    )

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(dataset_test, 1,
                                  sampler=sampler_test, drop_last=False,
                                  collate_fn = utils.collate_fn_nt2,
                                  num_workers=1)
    
    device = torch.device(args.device)
    model = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    model.eval()

    probability = []
    ground_truth = []
    g1 = []
    g2 = []
    policy = []
    len_g1 = []
    len_g2 = []
    couple_id = []
    ids = []
    gradcam_results = []

    for (rna1, rna2), target in data_loader_test:

        s = target[0]
        interacting = bool(s['interacting'])
        s_bbox = s['bbox']
        int_bbox = s['interaction_bbox']

        row_original = df_nt[(df_nt['gene1'] == s['gene1'])&(df_nt['gene2'] == s['gene2'])]
        row_swapped = df_nt[(df_nt['gene2'] == s['gene1'])&(df_nt['gene1'] == s['gene2'])]

        if len(row_original)>0:
            #assert len(row_original) == 1 #there are few duplicates
            row = row_original.iloc[0]
            swapped_genes = False
        elif len(row_swapped)>0:
            #assert len(row_swapped) == 1 #there are few duplicates
            row = row_swapped.iloc[0]
            swapped_genes = True
        else:
            raise NotImplementedError

        id_sample = row.couples
        policy_sample = row.policy
        couple_id_sample = row.couples_id

        rna1, rna2 = rna1.to(device), rna2.to(device)

        outputs = model(rna1, rna2)

        if interacting:
            outputs[:, 1].backward()
            x1 = int(int_bbox.x1-s_bbox.x1)
            x2 = int(int_bbox.x2-s_bbox.x1)
            y1 = int(int_bbox.y1-s_bbox.y1)
            y2 = int(int_bbox.y2-s_bbox.y1)
            width = s_bbox.x2-s_bbox.x1
            height = s_bbox.y2-s_bbox.y1
            gradcam_results.append(xai.get_gradcam_results(model, id_sample, swapped_genes, outputs, rna1, rna2, height, width, x1, x2, y1, y2, treshold = 75))

        probability += outputs.softmax(-1)[:, 1].tolist()

        ground_truth.append(1 if interacting else 0)
        policy.append(policy_sample)
        couple_id.append(couple_id_sample)
        g1.append(row['gene1'])
        g2.append(row['gene2'])
        len_g1.append(s['bbox'].x2 - s['bbox'].x1)
        len_g2.append(s['bbox'].y2 - s['bbox'].y1)
        # original_length1.append(or_len1)
        # original_length2.append(or_len2)
        ids.append(id_sample)

    res = pd.DataFrame({
        'id_sample':ids,
        'probability':probability,
        'ground_truth':ground_truth,
        'g1':g1,
        'g2':g2,
        'policy':policy,
        'len_g1': len_g1,
        'len_g2': len_g2,
        'couples':couple_id,
    })

    gradcam_results = pd.DataFrame(gradcam_results)

    res['prediction'] = (res['probability'] > 0.5).astype(int)
    res['sampled_area'] = res['len_g1']*res['len_g2']
    
    df = pd.read_csv(os.path.join(processed_files_dir,"final_df.csv"), sep = ',')[['couples', 'protein_coding_1', 'protein_coding_2', 'length_1', 'length_2']].rename({'length_1':'original_length1', 'length_2':'original_length2'}, axis = 1)
    assert df.merge(res, on = 'couples').shape[0] >= res.shape[0]
    if df.merge(res, on = 'couples').shape[0] > res.shape[0]:
        print(f"Be careful, some prediction will be counted more than one time. The number of duplicated sequences is {(df.merge(res, on = 'couples').drop_duplicates().shape[0]-res.shape[0])}")

    res = df.merge(res, on = 'couples').drop_duplicates().reset_index(drop = True)
    res=res.rename({'protein_coding_1': 'gene1_pc'}, axis = 1)
    res=res.rename({'protein_coding_2': 'gene2_pc'}, axis = 1)

    df_genes_original = pd.read_csv(os.path.join(processed_files_dir,"df_genes.csv"), sep = ',')[['gene_id', 'species_set']].rename({'gene_id':'original_gene_id'}, axis = 1)
    df_genes = df_genes_nt[['gene_id', 'original_gene_id']].merge(df_genes_original, on = 'original_gene_id')

    res = res.merge(df_genes, left_on = 'g1', right_on = 'gene_id').drop('gene_id', axis = 1).rename({'species_set':'specie'}, axis = 1)
    res['specie'] = res.specie.str.replace("{'hs'}", "human")
    res['specie'] = res.specie.str.replace("{'mm'}", "mouse")

    g12 = res.couples.str.extractall('(.*)_(.*)').reset_index(drop = True)
    res['gene1_original'], res['gene2_original'] = g12[0], g12[1]

    res.to_csv(os.path.join(checkpoint_dir, 'test_results.csv'))
    gradcam_results.to_csv(os.path.join(checkpoint_dir, 'gradcam_results.csv'))

    if RUN_FINETUNING:
        res.to_csv(os.path.join(checkpoint_dir, 'test_results_finetuning.csv'))
        gradcam_results.to_csv(os.path.join(checkpoint_dir, 'gradcam_results_finetuning.csv'))
    else:
        res.to_csv(os.path.join(checkpoint_dir, 'test_results.csv'))
        gradcam_results.to_csv(os.path.join(checkpoint_dir, 'gradcam_results.csv'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python run_binary_cl_on_test2.py &> run_binary_cl_on_test2.out &

    
    MAX_N_GROUP_SIZE = 300 
     
    checkpoint_dir = os.path.join(ROOT_DIR, 'checkpoints', 'binary_cl2')

    # Define the path to the file containing the args namespace
    args_path = os.path.join(checkpoint_dir, 'args.pkl')

    # Load the args namespace from the file
    with open(args_path, 'rb') as f:
        args_dict = pickle.load(f)

    # Convert the dictionary to an argparse.Namespace object
    args = argparse.Namespace(**args_dict)

    if RUN_FINETUNING:
        args.resume = os.path.join(args.output_dir, 'best_model_fine_tuning.pth') 
    else:
        args.resume = os.path.join(args.output_dir, 'best_model.pth') 

    seed_everything(123)
    main(args)