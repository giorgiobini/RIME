import pandas as pd
import os
import time
import numpy as np
import pickle
import argparse
import torch
import sys
import datetime
import time
sys.path.insert(0, '..')
from config import *
import util.xai as xai
import util.misc as utils
from dataset.data import (
    seed_everything,
)
from models.nt_classifier import build as build_model 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, processed_files_dir, original_files_dir, rna_rna_files_dir, metadata_dir, embedding_dir

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    parser.add_argument('--how', default='test',
                        help='Can be test or val')
    parser.add_argument('--dataset', default='paris',
                        help='Can be paris, mario, ricseq, splash')
    parser.add_argument('--folder', default='binary_cl2',
                        help='Default is binary_cl2')
    return parser

def main(args):

    dataset = args.dataset
    assert dataset in ['paris', 'mario', 'ricseq', 'splash']
    if dataset == 'paris':
        paris = True
    else:
        paris = False
    
    start_time = time.time() 

    if paris:
        df = pd.read_csv(os.path.join(metadata_dir, f'{HOW}500.csv'))
        df_nt = pd.read_csv(os.path.join(metadata_dir, 'df_nt_HQ.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_HQ.csv'))

    else:
        df = pd.read_csv(os.path.join(metadata_dir, f'{dataset}500.csv'))
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_{dataset}.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_{dataset}.csv'))


    
    device = torch.device(args.device)
    model = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    model.eval()

    folder_path = os.path.join(checkpoint_dir, f'gradcam_matrix_{dataset}')
    if not os.path.exists(folder_path):
        # Create the directory
        os.makedirs(folder_path)

    for _, row in df.iterrows():

        gene1, gene2, id_couple, l1, l2 = row['gene1'], row['gene2'], row['couples'], row['len1'], row['len2']
        
        df_nt_row = df_nt[df_nt.couples == id_couple].iloc[0]
        
        interacting = df_nt_row.interacting
        assert df_nt_row.gene1 == gene1
        assert df_nt_row.gene2 == gene2

        id_sample = id_couple

        if interacting:

            x1_emb, x2_emb, y1_emb, y2_emb = row.x1//6, row.x2//6, row.y1//6, row.y2//6

            embedding1_path = os.path.join(embedding_dir, '32', gene1+'.npy')
            embedding2_path = os.path.join(embedding_dir, '32', gene2+'.npy')
            
            embedding1 = np.load(embedding1_path)[x1_emb:x2_emb, :]
            embedding2 = np.load(embedding2_path)[y1_emb:y2_emb, :]
            rna1, rna2 =  torch.as_tensor(embedding1).unsqueeze(0), torch.as_tensor(embedding2).unsqueeze(0)
            rna1, rna2 = torch.transpose(rna1, 1, 2), torch.transpose(rna2, 1, 2)
            rna1, rna2 = rna1.to(device), rna2.to(device)
            
            outputs = model(rna1, rna2)


            outputs[:, 1].backward()
            x1 = int(row.seed_x1-row.x1)
            x2 = int(row.seed_x2-row.x1)
            y1 = int(row.seed_y1-row.y1)
            y2 = int(row.seed_y2-row.y1)
            width = row.len1
            height = row.len2

            probability = outputs.softmax(-1)[:, 1].tolist()[0]
            savepath = os.path.join(folder_path, str(id_sample) + '.pkl')

            xai.download_gradcam_matrixes(model, id_sample, rna1, rna2, height, width, x1, x2, y1, y2, probability, savepath = savepath)


        else:
            continue

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Downlaoding time {}'.format(total_time_str))
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python run_binary_cl_on_test_500_for_downloading_gradcam_matrixes.py &> run_binary_cl_on_test_500_for_downloading_gradcam_matrixes.out &
    #nohup python run_binary_cl_on_test_500_for_downloading_gradcam_matrixes.py --folder=binary_cl2 --how=test --dataset=paris &> run_binary_cl_on_test_500_for_downloading_gradcam_matrixes.out &
    #nohup python run_binary_cl_on_test_500_for_downloading_gradcam_matrixes.py --folder=binary_cl2 --dataset=mario &> run_binary_cl_on_test_mario500_for_downloading_gradcam_matrixes.out &
    #nohup python run_binary_cl_on_test_500_for_downloading_gradcam_matrixes.py --folder=binary_cl2 --dataset=ricseq &> run_binary_cl_on_test_ricseq500_for_downloading_gradcam_matrixes.out &
    #nohup python run_binary_cl_on_test_500_for_downloading_gradcam_matrixes.py --folder=binary_cl2 --dataset=splash &> run_binary_cl_on_test_splash500_for_downloading_gradcam_matrixes.out &
     


    parser = argparse.ArgumentParser('Test500', parents=[get_args_parser()])
    args = parser.parse_args()
    HOW = args.how 
    DATASET = args.dataset

    checkpoint_dir = os.path.join(ROOT_DIR, 'checkpoints', args.folder)

    # Define the path to the file containing the args namespace
    args_path = os.path.join(checkpoint_dir, 'args.pkl')

    # Load the args namespace from the file
    with open(args_path, 'rb') as f:
        args_dict = pickle.load(f)
    # Convert the dictionary to an argparse.Namespace object
    args = argparse.Namespace(**args_dict)

    args.dataset = DATASET

    args.resume = os.path.join(checkpoint_dir, 'best_model.pth') 

    seed_everything(123)
    main(args)