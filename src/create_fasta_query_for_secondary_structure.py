import sys
import os
import time
import argparse
from Bio import SeqIO
import pandas as pd
import numpy as np
sys.path.insert(0, '..')
import dataset.annotation_files_utils as utils

def get_args_parser(default_output_dir):
    parser = argparse.ArgumentParser('Set ufold query args', add_help=False)
    
    parser.add_argument('--dataset', default='paris', type=str, 
                        help="Can be: {'paris', 'inference'}")

    parser.add_argument('--results_dir', default=default_output_dir, type=str, 
                        help="Where to save the output chunks")
    
    parser.add_argument('--input_dir', default='', type=str, 
                    help="Needed only if dataset == 'inference' ")
    
    return parser

def parse_gene_fasta(file_dir):
    d = {}
    file = SeqIO.parse(open(file_dir),'fasta')
    for i, fasta in enumerate(file):
        name = str(fasta.description)
        seq = str(fasta.seq)
        d[i] = {'gene_id': name, 
               'cdna':seq}
    return pd.DataFrame.from_dict(d, orient = 'index')

def main(args, df_cdna):
    
    output_file_dir = args.results_dir
    
    interval_len = 600
    
    seqs_per_file = 1000
    list_df = [df_cdna[i:i+seqs_per_file] for i in range(0,df_cdna.shape[0],seqs_per_file)]

    for num, df_split in enumerate(list_df):
        output_file = os.path.join(output_file_dir, 'chunk{}.txt'.format(str(num)))
        for j, x in df_split.iterrows():
            list_seq = []
            list_name = []
            gene_id = x.gene_id
            seq = x.cdna.replace('T', 'U')
            seq_len = len(seq)
            n_runs = max(int(np.round(seq_len/interval_len)), 1)
            
            intervals = []
            linspace = np.linspace(0, seq_len, n_runs + 1).astype(int)
            for i in range(len(linspace) - 1):
                intervals.append((linspace[i], linspace[i+1]))
                
            for n, interval in enumerate(intervals):
                list_seq.append(seq[interval[0]:interval[1]])
                name = gene_id + '_' + str(n)
                list_name.append(name)

            fastafile = open(output_file, "a")
            for i in range(len(list_seq)):
                fastafile.write(">" + list_name[i] + "\n" +list_seq[i] + "\n")
            fastafile.close()

if __name__ == '__main__':
    #run me with: -> nohup python create_fasta_query_for_secondary_structure.py &> create_fasta_query_for_secondary_structure.out &
    ROOT_DIR = os.path.dirname(os.path.abspath('.'))
    original_files_dir = os.path.join(ROOT_DIR, 'dataset', 'original_files')
    processed_files_dir = os.path.join(ROOT_DIR, 'dataset', 'processed_files')
    default_output_dir = os.path.join(ROOT_DIR, 'UFold_dependencies', 'data')
    start_time = time.time()
    
    parser = argparse.ArgumentParser('Fasta for UFold query', parents=[get_args_parser(default_output_dir)])
    args = parser.parse_args()

    if os.path.isdir(args.results_dir) == False:
        os.mkdir(args.results_dir)
    
    if args.dataset == 'paris':
        df_cdna = utils.get_df_cdna(original_files_dir, processed_files_dir)
        df_cdna = df_cdna[['gene_id', 'cdna']]
    elif args.dataset == 'inference':
        input_dir = args.input_dir
        df_cdna = parse_gene_fasta(os.path.join(input_dir,'genes.fa'))
    else:
        raise AttributeError
        
    main(args, df_cdna)
    
    total_seconds = time.time() - start_time
    total_minutes = total_seconds/60
    print('Done in {} minutes'.format(total_minutes))
