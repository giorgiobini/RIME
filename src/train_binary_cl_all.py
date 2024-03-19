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
from torch.utils.data import DataLoader, ConcatDataset, Dataset
sys.path.insert(0, '..')
from util.engine import train_one_epoch
from util.engine import evaluate
from models.nt_classifier import build as build_model 
import util.misc as utils
import json
from dataset.data import (
    RNADataset,
    RNADatasetNT,
    RNADatasetNT500,
    EasyPosAugment,
    InteractionSelectionPolicy,
    SmartNegAugment,
    seed_everything,
)

INCLUDE_RICSEQ = False

from train_binary_cl import obtain_train_dataset_paris, obtain_policies_object, obtain_dataset_object, obtain_val_dataset_paris, obtain_train_dataset, obtain_val_dataset, seed_worker, get_args_parser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, processed_files_dir, original_files_dir, rna_rna_files_dir, metadata_dir, embedding_dir

# Custom dataset class to undersample the first two datasets
class UndersampledDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = min(len(dataset) for dataset in datasets)

    def __len__(self):
        return self.length * len(self.datasets)

    def __getitem__(self, index):
        dataset_idx = index // self.length
        sample_idx = index % self.length
        return self.datasets[dataset_idx][sample_idx]


def main(args):

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    if os.path.isfile(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    if INCLUDE_RICSEQ:
        dataset_ricseq, policies_ricseq = obtain_train_dataset('ricseq', EASY_PRETRAINING, TRAIN_HQ, FINETUNING, args.min_n_groups_train, args.max_n_groups_train, SPECIE)
    dataset_splash, policies_splash = obtain_train_dataset('splash', EASY_PRETRAINING, TRAIN_HQ, FINETUNING, args.min_n_groups_train, args.max_n_groups_train, SPECIE)
    dataset_paris, policies_paris = obtain_train_dataset('paris', EASY_PRETRAINING, TRAIN_HQ, FINETUNING, args.min_n_groups_train, args.max_n_groups_train, SPECIE)
    
    args.policies_train = policies_paris

    dataset_val, policies_val = obtain_val_dataset(VAL_DATASET, EASY_PRETRAINING, FINETUNING, args.min_n_groups_val, args.max_n_groups_val, SPECIE)
    args.policies_val = policies_val
    
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)


     # Save the namespace of the args object to a file using pickle
    with open(os.path.join(args.output_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args.__dict__, f)

    
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
    for epoch in range(args.start_epoch, args.epochs):
        
        if utils.early_stopping(n_epochs = args.n_epochs_early_stopping, 
                                current_epoch = epoch, 
                                best_model_epoch = utils.best_model_epoch(output_dir / "log.txt")):
            break
            
            
        # Create UndersampledDataset with the concatenated dataset
        if INCLUDE_RICSEQ:
            undersampled_dataset = UndersampledDataset([dataset_ricseq, dataset_splash, dataset_paris])
        else:
            undersampled_dataset = UndersampledDataset([dataset_splash, dataset_paris])
        
        # Create DataLoader using the undersampled dataset
        data_loader_train = DataLoader(undersampled_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=utils.collate_fn_nt2, num_workers=args.num_workers)
    
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, grad_accumulate = 1)
        lr_scheduler.step()
        
        #manually difine data_loader_val at each epoch such that the validation set is fixed
        g = torch.Generator()
        g.manual_seed(23)
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

        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
                
        best_model_epoch = utils.best_model_epoch(os.path.join(output_dir, "log.txt"), metric = 'accuracy', maximize = True)

        best_loss = utils.best_model_epoch(log_path = os.path.join(output_dir, "log.txt"), metric = 'loss', maximize = False)
        best_recall = utils.best_model_epoch(log_path = os.path.join(output_dir, "log.txt"), metric = 'recall', maximize = True)
        best_specificty = utils.best_model_epoch(log_path = os.path.join(output_dir, "log.txt"), metric = 'specificity', maximize = True)
        
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs
        if (best_loss == epoch)|(best_recall == epoch)|(best_specificty == epoch)|(best_model_epoch == epoch):
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
            
        dataset_splash, policies_splash = obtain_train_dataset('splash', EASY_PRETRAINING, TRAIN_HQ, FINETUNING, args.min_n_groups_train, args.max_n_groups_train, SPECIE)
        dataset_paris, policies_paris = obtain_train_dataset('paris', EASY_PRETRAINING, TRAIN_HQ, FINETUNING, args.min_n_groups_train, args.max_n_groups_train, SPECIE)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    
if __name__ == '__main__':
    #run me with: -> 

    #nohup python train_binary_cl_all.py --val_dataset=splash &> train_binary_cl_all.out &

    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints', 'binary_cl2')
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')

    EASY_PRETRAINING = args.easy_pretraining
    FINETUNING = args.finetuning
    TRAIN_HQ = args.train_hq    
    SPECIE = args.specie
    TRAIN_DATASET = args.train_dataset
    VAL_DATASET = args.val_dataset

    if args.modelarch == 1:
        args.use_projection_module = True

    if args.use_projection_module == False:
        args.proj_module_N_channels = 0 # In this way I will not count the parameters of the projection module when I define the n_parameters variable

    seed_everything(123)
    main(args)