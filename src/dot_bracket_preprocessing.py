import os
import time
import pandas as pd
import argparse

def get_args_parser(default_files_dir, default_results_dir):
    parser = argparse.ArgumentParser('Set ufold args', add_help=False)
    
    parser.add_argument('--files_dir', default=default_files_dir, type=str, 
                        help="Where are the input chunks")

    parser.add_argument('--results_dir', default=default_results_dir, type=str, 
                        help="Where to save the output")
    return parser

def main(input_file_dir, output_file):
    input_file_list = [os.path.join(input_file_dir, chunk) for chunk in os.listdir(input_file_dir) if chunk.startswith('chunk')] #put here all the fasta chunks #os.path.join(input_file_dir, 'input_dot_ct_file_old.txt')

    dfs = []

    for input_file in input_file_list:
        
        subset = pd.read_csv(input_file, sep = '\t', header = None)
        subset.rename({0:'genes', 1:'dot_br'}, axis = 1, inplace=True)
        rgx = subset.genes.str.extractall(r"(.*)_(.*)").reset_index()
        subset['genes'] = rgx[0]
        subset['numbers'] = rgx[1]
        dfs.append(subset)

    df = pd.concat(dfs, axis = 0)

    del dfs
    del subset

    df['numbers']= df['numbers'].astype(int)

    df = df.sort_values('numbers', ascending=True).reset_index(drop = True)
    #print(df[df.genes=='ENSMUSG00000109564'])
    #df = df.astype({"numbers": str}).groupby(['genes'], as_index = False).agg({'numbers': ''.join})
    df = df[['genes', 'dot_br']].groupby(['genes'], as_index = False).agg({'dot_br': ''.join})
    df.to_csv(output_file, sep='\t', index_label = False)
    
if __name__ == '__main__':
    #run me with: -> nohup python dot_bracket_preprocessing.py &> dot_bracket_preprocessing.out &
    ROOT_DIR = os.path.dirname(os.path.abspath('.'))
    processed_files_dir = os.path.join(ROOT_DIR, 'dataset', 'processed_files')
    
    default_files_dir = os.path.join(ROOT_DIR, 'UFold_dependencies', 'results')
    default_results_dir = processed_files_dir
    
    start_time = time.time()
    
    parser = argparse.ArgumentParser('UFold', parents=[get_args_parser(default_files_dir, default_results_dir)])
    args = parser.parse_args()
    
    files_dir = args.files_dir
    results_dir = args.results_dir
    output_file = os.path.join(results_dir, 'dot_bracket.txt')
    
    if os.path.isdir(results_dir) == False:
        os.mkdir(results_dir)
    
    main(files_dir, output_file)
    
    total_seconds = time.time() - start_time
    total_minutes = total_seconds/60
    print('Done in {} minutes'.format(total_minutes))