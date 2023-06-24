import pandas as pd
import os
import time
import numpy as np
import argparse
import torch
import datetime
import sys
import random
from pathlib import Path
sys.path.insert(0, '..')
from util.engine import train_one_epoch_binary_cl as train_one_epoch
from util.engine import evaluate_binary_cl as evaluate
import util.contact_matrix as cm
from models.binary_classifier import build as build_model 
import util.misc as utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, processed_files_dir, original_files_dir, rna_rna_files_dir, metadata_dir, embedding_dir

def main():
    df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt.csv'))
    for _, row in df_genes_nt.iterrows():
        embedding_path = os.path.join(embedding_dir, '32', row.gene_id + '.npy')
        try:
            np.load(embedding_path)
        except:
            with open('/data01/giorgio/RNARNA-NT/src/error.txt', 'a') as f:
                print(embedding_path)
                f.write(embedding_path + '\n')
                f.close()
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python problematic_embedding_DROP.py &> problematic_embedding_DROP.out &

    main()