import os 
import time
from torch.utils import data
import argparse
import pandas as pd
import sys

def get_args_parser(default_files_dir, default_results_dir):
    parser = argparse.ArgumentParser('Set ufold args', add_help=False)
    
    parser.add_argument('--files_dir', default=default_files_dir, type=str, 
                        help="Where are the input chunks")

    parser.add_argument('--results_dir', default=default_results_dir, type=str, 
                        help="Where to save the output chunks")
    
    parser.add_argument('--num_workers', default=10, type=int, 
                        help="Number of workers")
    
    parser.add_argument('--device', default = 'cuda:0', type=str, 
                        help="Torch device")
    return parser


def main(files_dir, results_dir, num_workers, device):

    chunks = [file for file in os.listdir(files_dir) if file.startswith('chunk')]

    UFoldFeatureExtractor = UFoldModel(device, ufold_path, eval_mode = True)

    for chunk in chunks:
        start_time_chunk = time.time()

        chunk_name = os.path.splitext(chunk)[0]

        test_data = RNASSDataGenerator_input(files_dir, chunk_name)

        params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': num_workers,
                'drop_last': True}

        test_set = Dataset_FCN(test_data)
        test_generator = data.DataLoader(test_set, **params)

        results_in_batch_1 = {}
        for batch, (seq_embeddings, seq_lens, seq_ori, seq_name) in enumerate(test_generator):
            model_features = UFoldFeatureExtractor.contact_net_prediction(seq_embeddings, seq_lens[0], seq_ori, seq_name, postprocessing = True)
            name = seq_name[0]
            one_hot = from_matrix_to_one_hot(model_features.squeeze(), device = device)
            dot_torch = from_one_hot_to_dot(one_hot)
            results_in_batch_1[name] = {'dot_seq':dot_torch}

        result_path = os.path.join(results_dir, chunk)

        pd.DataFrame.from_dict(results_in_batch_1, orient = 'index').to_csv(result_path, sep = '\t', header = False)

        total_minutes_chunk = (time.time() - start_time_chunk)/60
        print('{} done in {} minutes'.format(chunk_name, total_minutes_chunk))

if __name__ == '__main__':
    #run me with: -> nohup python run_ufold_predictions.py &> run_ufold_predictions.out &
    ROOT_DIR = os.path.dirname(os.path.abspath('.'))
    ufold_path= os.path.join(ROOT_DIR, 'UFold_dependencies', 'models', 'ufold_train_alldata.pt')
    
    default_files_dir = os.path.join(ROOT_DIR, 'UFold_dependencies', 'data')
    default_results_dir = os.path.join(ROOT_DIR, 'UFold_dependencies', 'results')
    
    start_time = time.time()
    
    parser = argparse.ArgumentParser('UFold', parents=[get_args_parser(default_files_dir, default_results_dir)])
    args = parser.parse_args()
    
    files_dir = args.files_dir
    results_dir = args.results_dir
    device = args.device
    num_workers = args.num_workers
    
    sys.argv = [sys.argv[0]] #need this before all the other imports
    
    from running_ufold import *
    from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
    from ufold.data_generator import RNASSDataGenerator_input
    #from running_ufold import from_matrix_to_one_hot, from_matrix_to_one_hot, UFoldModel
    
    if os.path.isdir(results_dir) == False:
        os.mkdir(results_dir)
    
    main(files_dir, results_dir, num_workers, device)
    
    total_seconds = time.time() - start_time
    total_minutes = total_seconds/60
    print('Done in {} minutes'.format(total_minutes))