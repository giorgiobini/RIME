import pandas as pd
import os
import time
import numpy as np
import argparse
import torch
import datetime
import sys
import pickle
import random
import shutil
from pathlib import Path
sys.path.insert(0, '..')
from util.engine import train_one_epoch
from util.engine import evaluate
import util.contact_matrix as cm
from models.bert_classifier2 import build as build_model 
import util.misc as utils
import json
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, processed_files_dir, rna_rna_files_dir, dataset_files_dir, ufold_path, bert_pretrained_dir, MAX_RNA_SIZE_BERT

def load_data_bert_script(dataset_files_dir, MAX_RNA_SIZE_BERT):
    source_file_path = os.path.join(dataset_files_dir, 'data.py')
    data_bert_file_path = os.path.join(dataset_files_dir, 'data_bert.py')
    shutil.copy(source_file_path, data_bert_file_path)
    new_max_rna_length = MAX_RNA_SIZE_BERT

    # Read the content of data_bert.py
    with open(data_bert_file_path, 'r') as file:
        lines = file.readlines()

    # Find and modify the MAX_RNA_LENGTH line
    modified_lines = []
    for line in lines:
        if line.startswith('MAX_RNA_LENGTH'):
            modified_lines.append(f'MAX_RNA_LENGTH = {new_max_rna_length}\n')
        else:
            modified_lines.append(line)

    # Write the modified content back to data_bert.py
    with open(data_bert_file_path, 'w') as file:
        file.writelines(modified_lines)

load_data_bert_script(dataset_files_dir, MAX_RNA_SIZE_BERT)

from dataset.data_bert import (
    RNADataset,
    EasyPosAugment,
    RegionSpecNegAugment,
    InteractionSelectionPolicy,
    EasyNegAugment,
    SmartNegAugment,
    HardPosAugment,
    HardNegAugment,
    seed_everything,
)

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
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    
    # Projection module
    parser.add_argument('--proj_module_N_channels', default=32, type=int,
                        help="Number of channels of the projection module for bert")
    parser.add_argument('--proj_module_secondary_structure_N_channels', default=4, type=int,
                        help="Number of channels of the projection module for the secondary structure")
    parser.add_argument('--drop_secondary_structure', type=str_to_bool, nargs='?', const=True, default=False,
                        help="If True, the architecture will remain the same, but the secondary structure tensors will be replaced by tensors of zeros (you will have unuseful weights in the model).")
    parser.add_argument('--use_projection_module', type=str_to_bool, nargs='?', const=True, default=False,
                        help="If True, I will project the embeddings in a reduced space.")

    # * Model
    parser.add_argument('--dropout_prob', default=0.01, type=float,
                         help="Dropout in the MLP model")
    parser.add_argument('--num_hidden_layers', default=0, type=int,
                        help="Number of hidden layers in the MLP. The number of total layers will be num_hidden_layers+1")
    parser.add_argument('--dividing_factor', default=20, type=int,
                        help="If the input is 5120, the first layer of the MLP is 5120/dividing_factor")
    parser.add_argument('--output_channels_mlp', default=100, type=int,
                        help="The number of channels after mlp processing")
    parser.add_argument('--n_channels1_cnn', default=100, type=int,
                    help="Number of hidden channels (1 layer) in the final cnn")
    parser.add_argument('--n_channels2_cnn', default=150, type=int,
                    help="Number of hidden channels (2 layer) in the final cnn")

    # dataset policies parameters
    parser.add_argument('--min_n_groups_train', default=5, type=int,
                       help='both rna will be dividend in n_groups and averaged their values in each group. The n_groups variable is sampled in the range [min_n_groups, max_n_groups] where both extremes of the interval are included')
    parser.add_argument('--max_n_groups_train', default=80, type=int,
                       help='both rna will be dividend in n_groups and averaged their values in each group. The n_groups variable is sampled in the range [min_n_groups, max_n_groups] where both extremes of the interval are included')
    parser.add_argument('--min_n_groups_val', default=80, type=int,
                       help='both rna will be dividend in n_groups and averaged their values in each group. The n_groups variable is sampled in the range [min_n_groups, max_n_groups] where both extremes of the interval are included')
    parser.add_argument('--max_n_groups_val', default=80, type=int,
                       help='both rna will be dividend in n_groups and averaged their values in each group. The n_groups variable is sampled in the range [min_n_groups, max_n_groups] where both extremes of the interval are included')
    parser.add_argument('--policies_train', default='',
                        help='policies for training dataset')
    parser.add_argument('--policies_val', default='',
                        help='policies for validation dataset')

    # dataset parameters
    parser.add_argument('--dataset_path', default = '')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing') # originally was 'cuda'
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    
    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--n_epochs_early_stopping', default=50)
    return parser

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def main(args):

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    if os.path.isfile(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        
        
    df_genes = pd.read_csv(os.path.join(processed_files_dir, "df_genes.csv"))
    df = pd.read_csv(os.path.join(processed_files_dir, "final_df.csv"))
    #the N in the cdna creates problem with the tokenizer
    df_genes['problematic_set'] = df_genes['cdna'].apply(lambda x: False if (set(x) - set({'A', 'C', 'G', 'T'}) == set()) else True)
    genesN = set(df_genes[df_genes.problematic_set].gene_id.values)
    print(f'{len(genesN)} genes have N, so will be excluded')
    df = df[~(df.gene1.isin(genesN))|(df.gene2.isin(genesN))].reset_index(drop = True)

    subset_train = os.path.join(rna_rna_files_dir, f"gene_pairs_training.txt")
    subset_val = os.path.join(rna_rna_files_dir, f"gene_pairs_val.txt")
    
    #-----------------------------------------------------------------------------------------
    pos_width_multipliers = {4: 0.1, 10: 0.15, 
                     14: 0.15, 17: 0.1, 
                     19: 0.3, 21: 0.2}
    neg_width_windows = {(50, 150): 0.05, (150, 170): 0.02,
                        (170, 260): 0.05, (260, 350): 0.15,
                        (350, 450): 0.28, (450, 511): 0.1,
                        (511, 512): 0.35}
    assert np.round(sum(pos_width_multipliers.values()), 4) == np.round(sum(neg_width_windows.values()), 4) == 1

    policies_train = [
            EasyPosAugment(
                per_sample=0.5,
                interaction_selection=InteractionSelectionPolicy.LARGEST,
                width_multipliers=pos_width_multipliers,
                height_multipliers=pos_width_multipliers,
            ),
            SmartNegAugment(
                per_sample=0.25,
                interaction_selection=InteractionSelectionPolicy.LARGEST,
                width_multipliers=pos_width_multipliers,
                height_multipliers=pos_width_multipliers,
            ),
            EasyNegAugment(
                per_sample=0.05,
                width_windows=neg_width_windows,
                height_windows=neg_width_windows,
            ),
            HardPosAugment(
                per_sample=0.1,
                interaction_selection=InteractionSelectionPolicy.RANDOM_ONE,
                min_width_overlap=0.3,
                min_height_overlap=0.3,
                width_multipliers=pos_width_multipliers,
                height_multipliers=pos_width_multipliers,
            ),
            HardNegAugment(
                per_sample=0.05,
                width_windows=neg_width_windows,
                height_windows=neg_width_windows,
            ),
    ]

    dataset_train = RNADataset(
        gene2info=df_genes,
        interactions=df,
        subset_file=subset_train,
        augment_policies=policies_train,
    )
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    max_multipliers = {10000: 1.}

    policies_val = [
        EasyPosAugment(
            per_sample=1,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=max_multipliers,
            height_multipliers=max_multipliers,
        ),  
        SmartNegAugment(
            per_sample=0.5,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=max_multipliers,
            height_multipliers=max_multipliers,
        ),
    ]
    
    dataset_val = RNADataset(
        gene2info=df_genes,
        interactions=df,
        subset_file=subset_val,
        augment_policies=policies_val,
    )

   #-----------------------------------------------------------------------------------------

    # Save the namespace of the args object to a file using pickle
    with open(os.path.join(args.output_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args.__dict__, f)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_bert, num_workers=args.num_workers)
    
    device = torch.device(args.device)
    model = build_model(args, bert_pretrained_dir)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    criterion = torch.nn.CrossEntropyLoss()
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        if utils.early_stopping(n_epochs = args.n_epochs_early_stopping, 
                                current_epoch = epoch, 
                                best_model_epoch = utils.best_model_epoch(output_dir / "log.txt")):
            break

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, grad_accumulate = 2)

        lr_scheduler.step()
        
        #manually difine data_loader_val at each epoch such that the validation set is fixed
        g = torch.Generator()
        g.manual_seed(0)
        data_loader_val = DataLoader(dataset_val, args.batch_size,
                                     sampler=sampler_val, drop_last=False,
                                     collate_fn=utils.collate_fn_bert,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker, 
                                     generator=g,)
        
        test_stats = evaluate(model, criterion, data_loader_val, device)   
            

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
                
        best_model_epoch = utils.best_model_epoch(output_dir / "log.txt")
        
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            
        if best_model_epoch == epoch:
            checkpoint_paths.append(output_dir / f'best_model.pth')
            
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))   
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python train_binary_cl_bert.py &> train_binary_cl_bert.out &
    #nohup python train_binary_cl_bert.py --drop_secondary_structure=true &> train_binary_cl_bert.out &

    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints_bert', 'binary_cl')
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')
    
    seed_everything(123)
    main(args)
