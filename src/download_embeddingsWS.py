seed = 123
import time
import pandas as pd
import numpy as np
import os
import argparse
import haiku as hk
import jax
import jax.numpy as jnp
import nucleotide_transformer
from nucleotide_transformer.mypretrained import get_pretrained_model
import time
import sys

##########
## we add this retry in case there is another submission occupying the GPU
MAX_RETRIES = 10
RETRY_DELAY = 300 

retries = 0
while retries < MAX_RETRIES:
    try:
        random_key = jax.random.PRNGKey(0)
        break
    except jaxlib.xla_extension.XlaRuntimeError as e:
        if "DNN library initialization failed" in str(e):
            retries += 1
            print(f"CUDA memory issue detected. Retry {retries}/{MAX_RETRIES} in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        else:
            raise
else:
    print("Max retries reached. Exiting...")
    sys.exit(1)
#####


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def infer(sequences, forward_fn, tokenizer, parameters, random_key):
    tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
    # Infer
    outs = forward_fn.apply(parameters, random_key, tokens)
    return outs, tokens

def save_embeddings_from_batch(save_path, embeddings, padding_mask, ids, save_mean = False):
    batch_size = embeddings.shape[0]
    
    for i in range(batch_size):
        sample_embeddings = embeddings[i] #(dim + n_padding_elements, 2560)
        sample_padding_mask = padding_mask[i]

        # Apply padding mask to remove padding elements
        masked_embeddings = sample_embeddings[sample_padding_mask]  #(dim, 2560)
        
        if save_mean:
            masked_embeddings_mean = jnp.mean(masked_embeddings, axis=0)
            np.save(os.path.join(save_path, f'{ids[i]}.npy'), masked_embeddings_mean)
        else:
            np.save(os.path.join(save_path, f'{ids[i]}.npy'), masked_embeddings)
        
def create_masked_embeddings(outs, layer, tokens, tokenizer):
    embeddings = outs[f'embeddings_{layer}']
    
    # Remove the CLS token and paddings
    embeddings = embeddings[:, 1:, :]
    padding_mask = tokens[:, 1:] != tokenizer.pad_token_id
    
    return embeddings, padding_mask

def get_args_parser():
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--batch_size', default=22, type=int,
                        help="Batch size for the NT model")
    parser.add_argument('--embedding_layer', default='32',
                        help="Which is the embedding layer you cutted the NT model")
    parser.add_argument('--path_to_embedding_query_dir', default='/data01/giorgio/RNARNA-NT/dataset/processed_files/nt_data/metadata',
                        help="Where is the embedding_query.csv file")
    parser.add_argument('--embedding_dir', default='/data01/giorgio/RNARNA-NT/dataset/processed_files/nt_data/embeddings',
                        help="Where is the embedding directory. If it doesn t exist, it will be created")
    parser.add_argument('--save_mean', default=0, type=int,
                        help="0 for False, 1 for True. If true, only the mean embedding is saved (2560 shaped vector)")
    return parser

def main(args):
    
    df = pd.read_csv(os.path.join(args.path_to_embedding_query_dir, 'embedding_query.csv'))

    embeddings_layers_to_save = (int(args.embedding_layer),)
    model_name = '2B5_multi_species'
    
    # Get pretrained model
    parameters, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        mixed_precision=False,
        embeddings_layers_to_save=embeddings_layers_to_save,
        attention_maps_to_save=(),
        max_positions=1000,
        chkpt_dir = os.path.join(nt_dir, 'checkpoints')
    )
    
    forward_fn = hk.transform(forward_fn)
    print('Model loaded')
    
    print(df.shape[0], 'sequences to download')
    
    n_batch = int(df.shape[0]/args.batch_size)
    slices = np.linspace(0, df.shape[0], n_batch, dtype = np.int64)
    
    start_time = time.time()
    for i in range(len(slices)-1):

        df_slice = df[slices[i]:slices[i+1]]
        ids = list(df_slice.id_query.values)
        sequences = list(df_slice.cdna.values)

        try:
            outs, tokens = infer(sequences, forward_fn, tokenizer, parameters, random_key)
            embeddings, padding_mask = create_masked_embeddings(outs, args.embedding_layer, tokens, tokenizer)
            del outs
            save_embeddings_from_batch(save_path, embeddings, padding_mask, ids, save_mean=args.save_mean)
            del embeddings
            del padding_mask

            with open(os.path.join(metadata_dir, f"done_{args.embedding_layer}.txt"), 'a') as f:
                for idx in ids:
                    f.write(str(idx) + '\n')
        except Exception as e:
            with open(os.path.join(metadata_dir, f"excluded_{args.embedding_layer}.txt"), 'a') as f:
                for idx in ids:
                    f.write(str(idx) + str(e) + '\n')

        if i%200 == 0:
            perc = np.round(i/len(slices) * 100, 2)
            print(f'{perc}% done in {(time.time()-start_time)/60} minutes')
    
    minutes = np.round((time.time()-start_time)/60, 2)
    hours = np.round(minutes/60, 2)
    days = np.round(hours/24, 2)
    print(f"Total time to process batch: {minutes} minutes, {hours} hours, {days} days")
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python download_embeddings.py &> download_embeddings.out &
    #nohup python download_embeddings.py --batch_size=2  &> download_embeddings.out & 
    #nohup python download_embeddings.py --save_mean=1 --path_to_embedding_query_dir=/data01/giorgio/RNARNA-NT/dataset/processed_files/nt_data/mean_embeddings --embedding_dir=/data01/giorgio/RNARNA-NT/dataset/processed_files/nt_data/mean_embeddings &> download_embeddings.out &

    parser = argparse.ArgumentParser('Download NT Embeddings', parents=[get_args_parser()])
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.embedding_dir):
        os.makedirs(args.embedding_dir)
    
    save_path = os.path.join(args.embedding_dir, args.embedding_layer)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    main(args)
