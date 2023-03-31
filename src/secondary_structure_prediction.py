import sys
import os
import torch
import time
import numpy as np
sys.path.insert(0, '..')
import dataset.annotation_files_utils as utils
import random
import warnings
import pandas as pd

def apply_fn(x, model, f, n):
    if x.name%100 == 0:
        print(np.round(x.name/n * 100))
        
    
    gene_id = x.gene_id
    seq = x.cdna.replace('T', 'U')
    
    seq_len = len(seq)
    interval_len = 700
    n_runs = max(int(np.round(seq_len/interval_len)), 1)
    
    intervals = []
    linspace = np.linspace(0, seq_len, n_runs + 1).astype(int)
    for i in range(len(linspace) - 1):
        intervals.append((linspace[i], linspace[i+1]))
        
    dot_string_fragments = []
    for interval in intervals:
        dot_string_fragments.append(obtain_dot_string(seq[interval[0]:interval[1]], gene_id, model))
        
    dot_string = ''.join(dot_string_fragments)
    
    f.write(' '.join([gene_id, dot_string]) + '\n')
    
def obtain_dot_string(seq, gene_id, model):
    seq_embeddings, seq_lens, seq_ori, seq_name = get_seq_embeddings_seq_ori(seq, gene_id)
    model_features = model.contact_net_prediction(seq_embeddings, seq_lens, seq_ori, seq_name)
    dot_string = from_matrix_to_dot(model_features)
    return dot_string

def main():
    device = 'cuda'
    df_cdna = utils.get_df_cdna(original_files_dir, processed_files_dir)
    df_cdna = df_cdna[['gene_id', 'cdna']]
    total_n = df_cdna.shape[0]
    UFoldFeatureExtractor = UFoldModel(device, ufold_path, eval_mode = True)
    dot_bracket_annotations_path = os.path.join(processed_files_dir, 'dot_bracket_file.txt')
    with open(dot_bracket_annotations_path, 'a') as f:
            df_cdna.apply(apply_fn, model = UFoldFeatureExtractor, f=f, n = total_n, axis = 1)


    """
    from multiprocessing import Pool
    from multiprocessing import Manager
    from functools import partial

    n_workers = 3 #os.cpu_count() - 2

    manager = Manager()
    q = manager.Queue() 
    pool = Pool(n_workers) # Create a multiprocessing Pool

    res = pool.map(partial(save_dot_bracket, model=UFoldFeatureExtractor), df_split)

    jobs = []
    for subset in np.array_split(df_cdna, n_workers):
        job = pool.apply_async(save_dot_bracket, (subset, UFoldFeatureExtractor, q)) #apply_async
        jobs.append(job)
    
    for job in jobs: 
        print(job.wait(10))

    # collect results from the workers through the pool result queue
    with open(dot_bracket_annotations_path, 'a') as f:
        for job in jobs: 
            f.write(str(job.get()) + '\n')

    pool.close()
    pool.join()
        
    def save_dot_bracket(df_cdna, model):
        print('ok')
        list_genes = []
        list_db = []
        for idx, row in df_cdna.iterrows():
            g, db = apply_fn(row, model=model)
            list_genes.append(g)
            list_db.append(db)
        return list_genes, list_db
        
    def apply_fn(x, model):
        gene_id = x.gene_id
        seq = x.cdna[:10].replace('T', 'U')
        seq_embeddings, seq_lens, seq_ori, seq_name = get_seq_embeddings_seq_ori(seq, gene_id)
        model_features = model.contact_net_prediction(seq_embeddings, seq_lens, seq_ori, seq_name)
        dot_string = from_matrix_to_dot(model_features)
        #line = ' '.join(gene_id, dot_string)
        return gene_id, dot_string

    def save_dot_bracket(x, model, file):
        gene_id = x.gene_id
        seq = x.cdna[:10].replace('T', 'U')
        seq_embeddings, seq_lens, seq_ori, seq_name = get_seq_embeddings_seq_ori(seq, gene_id)
        model_features = model.contact_net_prediction(seq_embeddings, seq_lens, seq_ori, seq_name)
        dot_string = from_matrix_to_dot(model_features)
        file.write(gene_id + ' ' + dot_string + '\n')
    """

if __name__ == '__main__':
    #run me with: -> nohup python secondary_structure_prediction.py &> secondary_structure.out &
    ROOT_DIR = os.path.dirname(os.path.abspath('.'))
    original_files_dir = os.path.join(ROOT_DIR, 'dataset', 'original_files')
    processed_files_dir = os.path.join(ROOT_DIR, 'dataset', 'processed_files')
    ufold_dir = os.path.join(ROOT_DIR, 'UFold_dependencies')
    ufold_path= os.path.join(ROOT_DIR, 'UFold_dependencies', 'models', 'ufold_train_alldata.pt')
    sys.path.insert(0, ufold_dir)
    from UFold_dependencies.running_ufold import get_seq_embeddings_seq_ori, UFoldModel, from_matrix_to_dot
    warnings.filterwarnings("ignore")
    start_time = time.time()
    main()
    total_seconds = time.time() - start_time
    total_minutes = total_seconds/60
    print('Done in {} minutes'.format(total_minutes))
