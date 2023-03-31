import pandas as pd
import os
import time
import numpy as np
from collections import Counter
import pickle
import argparse
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

sys.path.insert(0, '..')
from train_binary_cl import (
    get_args_parser,
    RNADataset,
    ROOT_DIR,
    EasyPosAugment,
    RegionSpecNegAugment,
    InteractionSelectionPolicy,
    EasyNegAugment,
    HardPosAugment,
    HardNegAugment,
    plot_sample,
    plot_sample2,
    seed_everything,
    pos_width_multipliers, 
    pos_height_multipliers, 
    neg_width_windows, 
    neg_height_windows,
)

from models.binary_classifier import build as build_model, obtain_predictions_ground_truth, calc_metrics
import util.misc as utils
import util.box_ops as box_ops


ROOT_DIR = os.path.dirname(os.path.abspath('.'))
original_files_dir = os.path.join(ROOT_DIR, 'dataset', 'original_files')
processed_files_dir = os.path.join(ROOT_DIR, 'dataset', 'processed_files')
rna_rna_files_dir = os.path.join(ROOT_DIR, 'dataset', 'rna_rna_pairs')
ufold_dir = os.path.join(ROOT_DIR, 'UFold_dependencies')
ufold_path= os.path.join(ufold_dir, 'models', 'ufold_train_alldata.pt')
bert_pretrained_dir = os.path.join(ROOT_DIR, 'dataset', 'pre_trained_DNABERT', '6-new-12w-0')

n_iters = 500
max_perturbations_per_iter = 1000


def main(args):
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_dir = Path(args.output_dir)

    if os.path.isfile(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    policies = [
        EasyPosAugment(
            per_sample=3,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=pos_width_multipliers,
            height_multipliers=pos_height_multipliers,
        ),
        EasyNegAugment(
            per_sample=3,
            width_windows=neg_width_windows,
            height_windows=neg_height_windows,
        ),
        HardPosAugment(
            per_sample=0.5,
            interaction_selection=InteractionSelectionPolicy.RANDOM_ONE,
            min_width_overlap=0.3,
            min_height_overlap=0.3,
            width_multipliers=pos_width_multipliers,
            height_multipliers=pos_height_multipliers,
        ),
        HardNegAugment(
            per_sample=0.5,
            width_windows=neg_width_windows,
            height_windows=neg_height_windows,
        ),
    ]
    
    dataset_test = RNADataset(
            gene_info_path=os.path.join(processed_files_dir, "df_cdna.csv"),
            interactions_path=os.path.join(processed_files_dir,"df_annotation_files_cleaned.csv"), #subset_valentino.csv
            dot_bracket_path=os.path.join(processed_files_dir,"dot_bracket.txt"),
            df_genes_path = os.path.join(processed_files_dir,"df_genes.csv"),
            subset_file = os.path.join(rna_rna_files_dir, "gene_pairs_val_random_filtered.txt"),
            augment_policies=policies
        )

    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args, bert_pretrained_dir, ufold_path)
    model.to(device)
    model_without_ddp = model

    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    
    model.eval()
    criterion.eval()

    filename = os.path.join(processed_files_dir, 'lime_results.txt')

    row = ['gene1', 'gene2', 'policy', 'x1', 'x2', 'y1', 'y2', 
           'gene1_specie', 'gene1_protein_coding', 'gene1_length', 
           'gene2_specie', 'gene2_protein_coding', 'gene2_length', 
           'int_box_x1', 'int_box_x2', 'int_box_y1', 'int_box_y2',
           'box_x1_hat', 'box_x2_hat', 'box_y1_hat', 'box_y2_hat',
           'cos_sim', 'cos_sim_random', 'score', 
           'iou_value', 'iou_random', 'probability', 'time'
          ]

    with open(f'{filename}', 'a') as the_file:
        line = ' '.join(row) + '\n'
        the_file.write(line)

    with torch.no_grad():
        for s in dataset_test:
            real = 1 if s['policy'] in ['hardpos', 'easypos'] else 0
            if real == 1:
                cdna1_slice, cdna2_slice = get_cdna_slices(s)
                probability = forward_func(cdna1_slice, cdna2_slice, s, model, UFoldFeatureExtractor, device)
                probability = float(np.round(probability.cpu(), 4))
                if (probability>0.5):
                    start_time = time.time()
                    
                    expl_matrix, score = lime(cdna1_slice, cdna2_slice, n_iters, max_perturbations_per_iter, probability, s, model, UFoldFeatureExtractor, device)
                    end_time = time.time()
                    total_minutes = np.round((end_time-start_time)/60, 2)
                    
                    expl_matrix_tr = expl_matrix_treshold(expl_matrix, treshold = 99, normalize = True)
                    s_bbox = s['sample_bbox']
                    int_bbox = s['interaction_bbox']
                    x1 = int_bbox.x1-s_bbox.x1
                    x2 = int_bbox.x2-s_bbox.x1
                    y1 = int_bbox.y1-s_bbox.y1
                    y2 = int_bbox.y2-s_bbox.y1
                    w = x2-x1
                    h = y2-y1
                    
                    cos_sim = float(np.round(cosine_similarity_expl(expl_matrix_tr, [x1, x2, y1, y2]), 3))
                
                    random_expl_matrix = np.random.rand(expl_matrix_tr.shape[0], expl_matrix_tr.shape[1])
                    random_expl_matrix = expl_matrix_treshold(random_expl_matrix, treshold = 99, normalize = True)
                    cos_sim_random = float(np.round(cosine_similarity_expl(random_expl_matrix, [x1, x2, y1, y2]), 3))
                    
                    x1hat, y1hat, what, hhat = estimate_bbox(expl_matrix_tr, desired_dim = 45)
                    iou_value = float(np.round(box_ops.iou_metric([x1hat, y1hat, what, hhat], 
                                              [x1, y1, w, h]), 
                           2))
                    random_x1, random_y1, random_w, random_h = estimate_bbox(random_expl_matrix, desired_dim = 45)
                    iou_random = float(np.round(box_ops.iou_metric([random_x1, random_y1, random_w, random_h], 
                                               [x1, y1, w, h]), 
                            2))
                    
                    row = [s['gene1'], s['gene2'], s['policy'], 
                           s['sample_bbox'].x1, s['sample_bbox'].x2, s['sample_bbox'].y1, s['sample_bbox'].y2,
                           s['gene1_info']['species'], s['gene1_info']['protein_coding'], s['gene1_info']['length'],
                           s['gene2_info']['species'], s['gene2_info']['protein_coding'], s['gene2_info']['length'],
                           s['interaction_bbox'].x1, s['interaction_bbox'].x2, s['interaction_bbox'].y1, s['interaction_bbox'].y2,
                           x1hat, y1hat, what, hhat,
                           cos_sim, cos_sim_random, score, 
                           iou_value, iou_random, probability, total_minutes]

                    row = [str(i) for i in row]

                    with open(f'{filename}', 'a') as the_file:
                        line = ' '.join(row) + '\n'
                        the_file.write(line)

    
if __name__ == '__main__':
    #run me with: -> nohup python xai_lime.py &> xai_lime.out &

    seed_everything(123)

    df = pd.read_csv(os.path.join(processed_files_dir,"df_cdna.csv"))
    
    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints', 'binary_cl')
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')
    args.device = 'cuda:1'
    args.resume = args.resume.replace('checkpoint.pth', 'best_model.pth')

    sys.path.insert(0, ufold_dir)
    from UFold_dependencies.running_ufold import UFoldModel
    from util.xai import lime, plot_lime_matrix, get_cdna_slices, forward_func, expl_matrix_treshold, cosine_similarity_expl, estimate_bbox
    UFoldFeatureExtractor = UFoldModel(args.device, ufold_path, eval_mode = True)
    main(args)
