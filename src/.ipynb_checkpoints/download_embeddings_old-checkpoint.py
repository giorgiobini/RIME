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
random_key = jax.random.PRNGKey(0)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def calculate_grouped_mean_embeddings(outs, layer, tokens, tokenizer, k):
    # Get the embeddings for the specified layer
    embeddings = outs[f"embeddings_{layer}"]
    
    # Remove the CLS token and paddings
    embeddings = embeddings[:, 1:, :]
    padding_mask = jnp.expand_dims(tokens[:, 1:] != tokenizer.pad_token_id, axis=-1)
    masked_embeddings = embeddings * padding_mask
    
    # Calculate the number of groups
    batch_size = masked_embeddings.shape[0]
    seq_length = masked_embeddings.shape[1]
    num_groups = seq_length // k
    
    # Reshape the embeddings to form groups
    grouped_embeddings = jnp.reshape(masked_embeddings[:, :num_groups*k, :], (batch_size, num_groups, k, -1))
    grouped_padding_mask = jnp.reshape(padding_mask[:, :num_groups*k, :], (batch_size, num_groups, k, -1))
    
    # Calculate the mean embeddings for each group
    group_sum_embeddings = jnp.sum(grouped_embeddings, axis=2)
    sequence_count_in_groups = grouped_padding_mask.sum(axis=2)
    sequence_count_in_groups = jnp.where(sequence_count_in_groups == 0, 1, sequence_count_in_groups)  # to avoid division by zero
    group_mean_embeddings = group_sum_embeddings / sequence_count_in_groups
    
    if k == 999:
        group_mean_embeddings = group_mean_embeddings.squeeze()
    
    return group_mean_embeddings

def infer(sequences, forward_fn, tokenizer, parameters, random_key):
    tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
    # Infer
    outs = forward_fn.apply(parameters, random_key, tokens)
    return outs, tokens

def save_data_to_folder(data, labels, ids, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i in range(len(data)):
        sample = data[i]
        label = labels[i]
        id_sample = ids[i]
        if label == 0:
            class_folder = os.path.join(folder_path, 'class_0')
        else:
            class_folder = os.path.join(folder_path, 'class_1')
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        np.save(os.path.join(class_folder, f'{id_sample}.npy'), sample)

def get_args_parser():
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--batch_size', default=13, type=int,
                        help="Batch size for the NT model")
    parser.add_argument('--k', default=999, type=int,
                        help="Which is the group size for the embeddings. If k = 999 Then I will have 1 group with the mean embedding of the entire sequence.")
    parser.add_argument('--set_data', default='training',
                        help='Can be training val, test') # originally was 'cuda'
    return parser

def main(args):
    set_data = args.set_data
    assert set_data in ['training', 'val', 'test']
    k = args.k
    k_dir = os.path.join(embedding_dir, str(k))
    if not os.path.exists(k_dir):
        os.makedirs(k_dir)
    batch_size = int(args.batch_size)
    
    meta = pd.read_csv(os.path.join(metadata_dir, f'{set_data}.csv'))
    embeddings_layers_to_save = (22, 30)

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
    
    n_batch = int(meta.shape[0]/batch_size)
    slices = np.linspace(0, meta.shape[0], n_batch, dtype = np.int64)
    
    start_time = time.time()
    for i in range(len(slices)-1):

        meta_slice = meta[slices[i]:slices[i+1]]
        labels = list(meta[slices[i]:slices[i+1]].interacting.values.astype(int))
        ids = list(meta[slices[i]:slices[i+1]].id_sample.values)

        try:
            sequences1 = list(meta_slice.cdna1.values)
            outs1, tokens1 = infer(sequences1, forward_fn, tokenizer, parameters, random_key)

            sequences2 = list(meta_slice.cdna2.values)
            outs2, tokens2 = infer(sequences2, forward_fn, tokenizer, parameters, random_key)

            for layer in embeddings_layers_to_save:
                layer_folder = os.path.join(k_dir, str(layer))
                if not os.path.exists(layer_folder):
                    os.makedirs(layer_folder)

                mean_embeddings1 = calculate_grouped_mean_embeddings(outs1, layer, tokens1, tokenizer, k) #shape is (batch_size, 2560)
                mean_embeddings2 = calculate_grouped_mean_embeddings(outs2, layer, tokens2, tokenizer, k) #shape is (batch_size, 2560)

                #concatenate the two embeddings (check if I am doing this properly, with the rigth axis)
                embeddings = np.concatenate((mean_embeddings1, mean_embeddings2), axis=1) #shape is (2*batch_size, 5120)

                #save the embeddings
                save_data_to_folder(embeddings, labels, ids, os.path.join(layer_folder, set_data))

            del outs1
            del outs2
        except:
            with open(os.path.join(metadata_dir, f"excluded_{set_data}.txt"), 'a') as f:
                for idx in ids:
                    f.write(str(idx) + '\n')

        with open(os.path.join(metadata_dir, f"done_{set_data}.txt"), 'a') as f:
            for idx in ids:
                f.write(str(idx) + '\n')

        if i%190 == 0:
            perc = np.round(i/len(slices) * 100, 2)
            print(f'{perc}% done in {(time.time()-start_time)/60} minutes')
    
    print(f"Total time to process batch: {(time.time()-start_time)/60} minutes")
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python download_embeddings.py &> download_embeddings.out &

    parser = argparse.ArgumentParser('Download NT Embeddings', parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)
