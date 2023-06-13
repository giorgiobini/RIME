import pandas as pd
import os
import time
import numpy as np
import argparse
import torch
import datetime
import sys
import random
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, '..')
from util.engine import train_one_epoch_mlp as train_one_epoch
from util.engine import evaluate_mlp as evaluate
from models.nt_classifier import build as build_model 
import util.misc as utils
import json
from dataset.data import (
    RNADataset,
    RNADatasetNT,
    EasyPosAugment,
    InteractionSelectionPolicy,
    SmartNegAugment,
    seed_everything,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, processed_files_dir, original_files_dir, rna_rna_files_dir, metadata_dir, embedding_dir

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    
    df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt.csv'))
    df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt.csv'))
    
    #-----------------------------------------------------------------------------------------
    subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_train_val_fine_tuning_nt.txt")

    with open(subset_train_nt, "rb") as fp:  # Unpickling
        list_train = pickle.load(fp)

    vc_train = df_nt[df_nt.couples.isin(list_train)].interacting.value_counts()
    assert vc_train[False]>vc_train[True]
    unbalance_factor = 1 - (vc_train[False] - vc_train[True]) / vc_train[False]

    pos_multipliers = {150: 0.3, 300: 0.3, 10_000_000: 0.4}
    neg_multipliers = pos_multipliers
    scaling_factor = 5

    policies_train = [
        EasyPosAugment(
            per_sample=1,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=pos_multipliers,
            height_multipliers=pos_multipliers,
        ),  
        SmartNegAugment(
            per_sample=unbalance_factor,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=neg_multipliers,
            height_multipliers=neg_multipliers,
        ),
    ] 
        
    dataset_train = RNADatasetNT(
            gene2info=df_genes_nt,
            interactions=df_nt,
            subset_file=subset_train_nt,
            augment_policies=policies_train,
            data_dir = os.path.join(embedding_dir, '32'),
            scaling_factor = scaling_factor,
            min_n_groups = args.min_n_groups_train,
            max_n_groups = args.max_n_groups_train,
    )
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_val_sampled_nt.txt")
    # subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_val_nt.txt")

    pos_multipliers = {10_000_000:1.,}
    neg_multipliers = pos_multipliers

    policies_val = [
        EasyPosAugment(
            per_sample=1,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=pos_multipliers,
            height_multipliers=pos_multipliers,
        ),  
        SmartNegAugment(
            per_sample=1, # unbalance_factor
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=neg_multipliers,
            height_multipliers=neg_multipliers,
        ),
    ]

    dataset_val = RNADatasetNT(
        gene2info=df_genes_nt,
        interactions=df_nt,
        subset_file=subset_val_nt,
        augment_policies=policies_val,
        data_dir = os.path.join(embedding_dir, '32'),
        scaling_factor = scaling_factor,
        min_n_groups = args.min_n_groups_val,
        max_n_groups = args.max_n_groups_val,
    )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    args.policies_val = policies_val
    args.policies_train = policies_train

   #-----------------------------------------------------------------------------------------
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_nt2, num_workers=args.num_workers)
    
    device = torch.device(args.device)
    model = build_model(args)
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
    for epoch in range(args.epochs):

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch)

        lr_scheduler.step()
        
        #manually difine data_loader_val at each epoch such that the validation set is fixed
        g = torch.Generator()
        g.manual_seed(0)
        data_loader_val = DataLoader(dataset_val, args.batch_size,
                                     sampler=sampler_val, drop_last=False,
                                     collate_fn=utils.collate_fn_nt2,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker, 
                                     generator=g,)
        
        test_stats = evaluate(model, criterion, data_loader_val, device)   
            

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        with (output_dir / "log_fine_tuning.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
                
        best_model_epoch = utils.best_model_epoch(output_dir / "log_fine_tuning.txt")
        
        checkpoint_paths = [output_dir / 'checkpoint_fine_tuning.pth']
            
        if best_model_epoch == epoch:
            checkpoint_paths.append(output_dir / f'best_model_fine_tuning.pth')
            
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
    print('Fine tuning time {}'.format(total_time_str))
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python train_binary_cl2_finetuning.py &> train_binary_cl2_finetuning.out &

    checkpoint_dir = os.path.join(ROOT_DIR, 'checkpoints', 'binary_cl2')

    # Define the path to the file containing the args namespace
    args_path = os.path.join(checkpoint_dir, 'args.pkl')

    # Load the args namespace from the file
    with open(args_path, 'rb') as f:
        args_dict = pickle.load(f)

    # Convert the dictionary to an argparse.Namespace object
    args = argparse.Namespace(**args_dict)

    if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint_fine_tuning.pth')):
        args.resume = os.path.join(checkpoint_dir, 'checkpoint_fine_tuning.pth')
    else:
        args.resume = os.path.join(args.output_dir, 'best_model.pth') 

    args.lr = 1e-5
    args.epochs = 5

    seed_everything(123)
    main(args)