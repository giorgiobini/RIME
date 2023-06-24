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
sys.path.insert(0, '..')
from util.engine import train_one_epoch_binary_cl as train_one_epoch
from util.engine import evaluate_binary_cl as evaluate
import util.contact_matrix as cm
from models.binary_classifier import build as build_model 
import util.misc as utils
import json
from torch.utils.data import DataLoader, DistributedSampler
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
    
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--lr_backbone', default=5e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    
    # Projection module
    parser.add_argument('--proj_module_N_channels', default=256, type=int,
                        help="Number of channels of the projection module for bert")
    parser.add_argument('--proj_module_secondary_structure_N_channels', default=4, type=int,
                        help="Number of channels of the projection module for the secondary structure")
    parser.add_argument('--drop_secondary_structure', type=str_to_bool, nargs='?', const=True, default=False,
                        help="If True, the architecture will remain the same, but the secondary structure tensors will be replaced by tensors of zeros (you will have unuseful weights in the model).")

    # * Backbone
    parser.add_argument('--backbone', default='mini_resnet18', type=str, 
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default = True, # action='store_true' (non default)
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--interediate_resnet_layer', default=3, type=int,
                        help="Where to cut the resnet model")
    parser.add_argument('--n_channels_backbone_out', default=64, type=int,
                        help="Number of channels out from the last layer (the one you cut)")
    parser.add_argument('--last_layer_intermediate_channels', default=64, type=int,
                        help="Number of intermediate channels in the last layer")

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=64, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=64, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.01, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=2, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', default = False) #parser.add_argument('--pre_norm', default = True) #parser.add_argument('--pre_norm', default = True, action='store_true')
    parser.add_argument('--build_decoder', default = False) 
    
    # * Binary Classification module
    parser.add_argument('--dropout_binary_cl', default=0.01, type=float,
                        help="Dropout applied in the Binary Classification module")

    # dataset policies parameters
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
    parser.add_argument('--n_epochs_early_stopping', default=20)
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

    
    df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt.csv'))
    df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt.csv'))
    
    #-----------------------------------------------------------------------------------------
    subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_training_nt.txt")

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
    )
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_val_nt.txt")
        
    with open(subset_val_nt, "rb") as fp:  # Unpickling
        list_val = pickle.load(fp)

    try:
        vc_val = df_nt[df_nt.couples.isin(list_val)].interacting.value_counts()
    except:
        vc_val = df_nt[df_nt.couple.isin(list_val)].interacting.value_counts() #I don t know the reason of this bug
    assert vc_val[True]>vc_val[False]
    unbalance_factor = 1 - (vc_val[True] - vc_val[False]) / vc_val[True]

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
            per_sample=unbalance_factor,
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
    )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    args.policies_val = policies_val
    args.policies_train = policies_train

   #-----------------------------------------------------------------------------------------

    # Save the namespace of the args object to a file using pickle
    with open(os.path.join(args.output_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args.__dict__, f)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_nt, num_workers=args.num_workers)
    
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
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

        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)

        lr_scheduler.step()
        
        #manually difine data_loader_val at each epoch such that the validation set is fixed
        g = torch.Generator()
        g.manual_seed(0)
        data_loader_val = DataLoader(dataset_val, args.batch_size,
                                     sampler=sampler_val, drop_last=False,
                                     collate_fn=utils.collate_fn_nt,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker, 
                                     generator=g,)
        
        test_stats = evaluate(model, criterion, postprocessors, data_loader_val, device, args.output_dir)   
            

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
    #nohup python train_binary_cl.py &> train_binary_cl.out &

    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints', 'binary_cl')
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')

    seed_everything(123)
    main(args)