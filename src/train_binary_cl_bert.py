import pandas as pd
import os
import time
import numpy as np
import argparse
import torch
import datetime
import sys
import random
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
    ROOT_DIR,
    EasyPosAugment,
    RegionSpecNegAugment,
    InteractionSelectionPolicy,
    EasyNegAugment,
    HardPosAugment,
    HardNegAugment,
    plot_sample,
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
    
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    
    # Projection module
    parser.add_argument('--proj_module_N_channels', default=16, type=int,
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
    parser.add_argument('--dropout', default=0.4, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=2, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', default = False) #parser.add_argument('--pre_norm', default = True) #parser.add_argument('--pre_norm', default = True, action='store_true')
    parser.add_argument('--build_decoder', default = False) 
    
    # * Binary Classification module
    parser.add_argument('--dropout_binary_cl', default=0.4, type=float,
                        help="Dropout applied in the Binary Classification module")

    # dataset parameters
    parser.add_argument('--dataset_path', default = '')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing') # originally was 'cuda'
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    
    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--n_epochs_early_stopping', default=10)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument("--local_rank", type=int, default=0)
    return parser

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_dir = Path(args.output_dir)

    if os.path.isfile(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')

    utils.init_distributed_mode(args)
    
    #-----------------------------------------------------------------------------------------
    pos_width_multipliers = {4: 0.05, 10: 0.1, 
                         14: 0.15, 17: 0.1, 
                         19: 0.3, 21: 0.3}
    pos_height_multipliers = pos_width_multipliers
    neg_width_windows = {(50, 150): 0.05, (150, 170): 0.12,
                         (170, 260): 0.05, (260, 350): 0.15,
                         (350, 450): 0.28, (450, 511): 0.1,
                         (511, 512): 0.25}
    neg_height_windows = neg_width_windows
    regionspec_multipliers = pos_width_multipliers
    regionspec_windows = neg_width_windows
    assert np.round(sum(pos_width_multipliers.values()), 4) == np.round(sum(neg_width_windows.values()), 4) == 1

    policies_train = [
            EasyPosAugment(
                per_sample=8,
                interaction_selection=InteractionSelectionPolicy.LARGEST,
                width_multipliers=regionspec_multipliers,
                height_multipliers=regionspec_multipliers,
            ),
            EasyNegAugment(
                per_sample=2,
                width_windows=neg_width_windows,
                height_windows=neg_height_windows,
            ),
            HardPosAugment(
                per_sample=3,
                interaction_selection=InteractionSelectionPolicy.RANDOM_ONE,
                min_width_overlap=0.3,
                min_height_overlap=0.3,
                width_multipliers=pos_width_multipliers,
                height_multipliers=pos_height_multipliers,
            ),
            HardNegAugment(
                per_sample=3,
                width_windows=neg_width_windows,
                height_windows=neg_height_windows,
            ),
            RegionSpecNegAugment(
            per_sample=7,
            width_windows=regionspec_multipliers,
            height_windows=neg_height_windows,
        ),
    ]
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    pos_width_multipliers_val = {3: 0.01, 19: 0.29, 21: 0.7}
    pos_height_multipliers_val = pos_width_multipliers_val
    neg_width_windows_val = {(50, 100): 0.05, (100, 160): 0.2, 
                             (160, 260): 0.05, (260, 350): 0.2, 
                             (350, 450): 0.25, (450, 511): 0.1, 
                             (511, 512): 0.15}
    neg_height_windows_val = neg_width_windows_val
    regionspec_multipliers_val = pos_width_multipliers_val
    regionspec_windows_val = neg_width_windows_val
    assert np.round(sum(pos_width_multipliers_val.values()), 4) == np.round(sum(neg_width_windows_val.values()), 4) == 1    
    policies_val = [
            EasyPosAugment(
                per_sample=6,
                interaction_selection=InteractionSelectionPolicy.LARGEST,
                width_multipliers=regionspec_multipliers_val,
                height_multipliers=regionspec_multipliers_val,
            ),
            EasyNegAugment(
                per_sample=4,
                width_windows=neg_width_windows_val,
                height_windows=neg_height_windows_val,
            ),
            HardPosAugment(
                per_sample=2,
                interaction_selection=InteractionSelectionPolicy.RANDOM_ONE,
                min_width_overlap=0.3,
                min_height_overlap=0.3,
                width_multipliers=pos_width_multipliers_val,
                height_multipliers=pos_height_multipliers_val,
            ),
            HardNegAugment(
                per_sample=4,
                width_windows=neg_width_windows_val,
                height_windows=neg_height_windows_val,
            ),
            RegionSpecNegAugment(
            per_sample=2,
            width_windows=regionspec_multipliers_val,
            height_windows=neg_height_windows_val,
        ),
    ]
   #-----------------------------------------------------------------------------------------
    
    
    dataset_train = RNADataset(
        gene_info_path=os.path.join(processed_files_dir, "df_cdna.csv"),
            interactions_path=os.path.join(
                processed_files_dir, "df_annotation_files_cleaned.csv"
            ),
            dot_bracket_path=os.path.join(processed_files_dir, "dot_bracket.txt"),
            subset_file=os.path.join(
                rna_rna_files_dir, "gene_pairs_training_random_filtered.txt"
            ),
        augment_policies=policies_train
    )

    
    dataset_val = RNADataset(
        gene_info_path=os.path.join(processed_files_dir, "df_cdna.csv"),
            interactions_path=os.path.join(
                processed_files_dir, "df_annotation_files_cleaned.csv"
            ),
            dot_bracket_path=os.path.join(processed_files_dir, "dot_bracket.txt"),
            subset_file=os.path.join(
                rna_rna_files_dir, "gene_pairs_val_random_filtered.txt"
            ),
        augment_policies=policies_val
    )

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args, bert_pretrained_dir, ufold_path)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
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
        
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        
        #manually difine data_loader_val at each epoch such that the validation set is fixed
        g = torch.Generator()
        g.manual_seed(0)
        data_loader_val = DataLoader(dataset_val, args.batch_size,
                                     sampler=sampler_val, drop_last=False,
                                     collate_fn=utils.collate_fn,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker, 
                                     generator=g,)
        
        test_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, device, args.output_dir
        )
            
            

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        best_model_epoch = utils.best_model_epoch(output_dir / "log.txt")
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                
            if best_model_epoch == epoch:
                checkpoint_paths.append(output_dir / f'best_model.pth')
                
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
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

    ROOT_DIR = os.path.dirname(os.path.abspath('.'))
    original_files_dir = os.path.join(ROOT_DIR, 'dataset', 'original_files')
    processed_files_dir = os.path.join(ROOT_DIR, 'dataset', 'processed_files')
    rna_rna_files_dir = os.path.join(ROOT_DIR, 'dataset', 'rna_rna_pairs')
    ufold_dir = os.path.join(ROOT_DIR, 'UFold_dependencies')
    ufold_path= os.path.join(ufold_dir, 'models', 'ufold_train_alldata.pt')
    bert_pretrained_dir = os.path.join(ROOT_DIR, 'dataset', 'pre_trained_DNABERT', '6-new-12w-0')

    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints', 'binary_cl')
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')
    
    seed_everything(123)
    main(args)
