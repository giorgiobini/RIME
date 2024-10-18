import pandas as pd
import os
import time
import numpy as np
import pickle
import argparse
from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt
import sys
import datetime
from pathlib import Path
import seaborn as sns
import time
from Bio import SeqIO
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc

sys.path.insert(0, '..')
from models.nt_classifier import build as build_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    parser.add_argument('--model_name', default='arch2_PARISfinetuned_PARIStest0023_PARISfinetunedFPweight_PARIStest0086',
                        help='Name of the model folder')
    parser.add_argument('--pairs_path', default='',
                        help='Path to the folder where there is pairs.csv')
    parser.add_argument('--device', default='cuda',
                        help='cuda or cpu')
    return parser

def main(args, device):

    df = pd.read_csv(os.path.join(pairs_path, 'pairs.csv'))
    df = df.filter(['id_pair','embedding1name','embedding2name','start_window1','end_window1','start_window2','end_window2'], axis = 1)
    print(df.shape[0], 'sequences to download')
    
    model = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.eval()
    
    
    
    start_time = time.time()
    
    ids = []
    probability = []
    for _, row in df.iterrows():
        id_pair, embedding1name, embedding2name, x1, x2, y1, y2 = row['id_pair'], row['embedding1name'], row['embedding2name'], row['start_window1'], row['end_window1'], row['start_window2'], row['end_window2']
        print(id_pair)
        if (x2-x1 >= 12)&(y2-y1 >= 12):
            x1_emb, x2_emb, y1_emb, y2_emb = x1//6, x2//6, y1//6, y2//6

            embedding1_path = os.path.join(embedding_dir, embedding1name+'.npy')
            embedding2_path = os.path.join(embedding_dir, embedding2name+'.npy')

            embedding1 = np.load(embedding1_path)[x1_emb:x2_emb, :]
            embedding2 = np.load(embedding2_path)[y1_emb:y2_emb, :]

            rna1, rna2 =  torch.as_tensor(embedding1).unsqueeze(0), torch.as_tensor(embedding2).unsqueeze(0)
            rna1, rna2 = torch.transpose(rna1, 1, 2), torch.transpose(rna2, 1, 2)
            rna1, rna2 = rna1.to(device), rna2.to(device)

            outputs = model(rna1, rna2)

            probability += outputs.softmax(-1)[:, 1].tolist()
            ids.append(id_pair)
        else:
            print(f'id: {id_pair} was not calculated because the window is too small')

    res = pd.DataFrame({
            'id_sample':ids,
            'probability':probability,
    })
    
    res.to_csv(os.path.join(pairs_path, 'predictions.csv'), index = False)
    
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python run_inference_new.py --pairs_path=/data01/giorgio/RNARNA-NT/dataset/external_dataset/check_predictions &> run_inference_new.out &

    parser = argparse.ArgumentParser('Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    model_name = args.model_name
    pairs_path = args.pairs_path
    #'/data01/giorgio/RNARNA-NT/dataset/external_dataset/check_predictions'

    embedding_dir = os.path.join(os.path.join(pairs_path, 'embeddings', '32')) #/data01/giorgio/RNARNA-NT/dataset/external_dataset/paris_windows_subset/embeddings/
    
    checkpoint_dir = os.path.join(ROOT_DIR, 'checkpoints', model_name)#'binary_cl2'
    
    device = torch.device(args.device)
    
    # Define the path to the file containing the args namespace
    args_path = os.path.join(checkpoint_dir, 'args.pkl')
    
    # Load the args namespace from the file
    with open(args_path, 'rb') as f:
        args_dict = pickle.load(f)

    # Convert the dictionary to an argparse.Namespace object
    args = argparse.Namespace(**args_dict)
    args.resume = os.path.join(checkpoint_dir, 'best_model.pth') 

    main(args, device)