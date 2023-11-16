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

def undersample_df(df, random_state = 23):
    neg = df[df.interacting == False]
    pos = df[df.interacting == True]
    if (pos.shape[0] > neg.shape[0]):
        return pd.concat([pos.sample(neg.shape[0], random_state = random_state), neg], axis = 0)
    elif (neg.shape[0] > pos.shape[0]):
        return pd.concat([neg.sample(pos.shape[0], random_state = random_state), pos], axis = 0)
    else:
        return df

def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Training mood

    parser.add_argument('--easy_pretraining', type=str_to_bool, nargs='?', const=True, default=False,
                        help="If True, I will train only for the interaction patches prediction task (no smartneg).")
    parser.add_argument('--finetuning', type=str_to_bool, nargs='?', const=True, default=False,
                        help="If True, train+val data are used for training")
    parser.add_argument('--train_hq', type=str_to_bool, nargs='?', const=True, default=False,
                        help="If True, only the good training positives (interaction length region >20) are used.")
    parser.add_argument('--specie', default='all',
                        help='can be all, human, mouse')
    
    # Projection module
    parser.add_argument('--proj_module_N_channels', default=128, type=int,
                        help="Number of channels of the projection module for nt")
    parser.add_argument('--proj_module_secondary_structure_N_channels', default=4, type=int,
                        help="Number of channels of the projection module for the secondary structure")
    parser.add_argument('--drop_secondary_structure', type=str_to_bool, nargs='?', const=True, default=False,
                        help="If True, the architecture will remain the same, but the secondary structure tensors will be replaced by tensors of zeros (you will have unuseful weights in the model).")
    parser.add_argument('--use_projection_module', type=str_to_bool, nargs='?', const=True, default=False,
                        help="If True, I will project the embeddings in a reduced space.")

    # * Model
    parser.add_argument('--modelarch', default=2, type=int,
                        help="Can be 1, 2. Architecture 1 has convolution projection on each branch and then contact matrix. Architecture 1 has mlp concatenation for each token combination and the contact matrix is built from this concat.")
    parser.add_argument('--dropout_prob', default=0.01, type=float,
                         help="Dropout in the MLP model")
    parser.add_argument('--args.mini_batch_size', default=32, type=int,
                        help="MLP batch size")
    parser.add_argument('--num_hidden_layers', default=0, type=int,
                        help="Number of hidden layers in the MLP. The number of total layers will be num_hidden_layers+1")
    parser.add_argument('--dividing_factor', default=10, type=int,
                        help="If the input is 5120, the first layer of the MLP is 5120/dividing_factor")
    parser.add_argument('--output_channels_mlp', default=800, type=int,
                        help="The number of channels after mlp processing")
    parser.add_argument('--n_channels1_cnn', default=300, type=int,
                    help="Number of hidden channels (1 layer) in the final cnn")
    parser.add_argument('--n_channels2_cnn', default=300, type=int,
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
    parser.add_argument('--n_epochs_early_stopping', default=100)
    return parser

def obtain_train_dataset(easy_pretraining, train_hq, finetuning, min_n_groups_train, max_n_groups_train, scaling_factor = 5):
    if easy_pretraining:
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_easy.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_easy.csv'))
        subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_training_nt_easy.txt")
    else:
        if train_hq:
            df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_HQ.csv'))
            df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_HQ.csv'))
            subset_train_nt = os.path.join(rna_rna_files_dir, 'gene_pairs_training_nt_HQ.txt')
        else:
            df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt.csv'))
            df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt.csv'))
            if finetuning:
                subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_train_val_fine_tuning_nt.txt")
            else:
                subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_training_nt.txt")

    with open(subset_train_nt, "rb") as fp:  # Unpickling
        list_train = pickle.load(fp)

    vc_train = df_nt[df_nt.couples.isin(list_train)].interacting.value_counts()
    assert vc_train[False]>vc_train[True]
    unbalance_factor = 1 - (vc_train[False] - vc_train[True]) / vc_train[False]

    if easy_pretraining:
        pos_multipliers = {15:0.2, 
                25:0.3,
                50:0.2, 
                100:0.23, 
                10_000_000: 0.07}

        neg_multipliers = {15:0.05, 
                        28:0.15,

                        40:0.08,
                        50:0.05,
                        60:0.1,

                        80:0.03,
                        90:0.03,
                        100:0.05,

                        110:0.05,

                        120:0.1,

                        140:0.05,
                        160:0.03,
                        180:0.03,
                        200:0.03,
                        220:0.02,
                        240:0.01,
                        260:0.01,

                        10_000_000: 0.1}
        
    else:
        pos_multipliers = {15:0.2, 
                       25:0.3,
                       50:0.2, 
                       100:0.23,
                       100_000_000:0.07}
        neg_multipliers = pos_multipliers

    policies_train = [
        EasyPosAugment(
            per_sample=0.5,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=pos_multipliers,
            height_multipliers=pos_multipliers,
        ),  
        SmartNegAugment(
            per_sample=unbalance_factor * 0.5,
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
            min_n_groups = min_n_groups_train,
            max_n_groups = max_n_groups_train,
    )
    
    return dataset_train, policies_train

def obtain_val_dataset(easy_pretraining, finetuning, min_n_groups_val, max_n_groups_val, scaling_factor = 5):

    if easy_pretraining:
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_easy.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_easy.csv'))
        subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_val_sampled_nt_easy.txt")
    else:
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_HQ.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_HQ.csv'))
        if finetuning:
            subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_test_sampled_nt_HQ.txt") # gene_pairs_test_sampled_nt.txt it is also HQ
            df500 = pd.read_csv(os.path.join(metadata_dir, f'test500.csv'))
        else:
            subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_val_sampled_nt_HQ.txt") # gene_pairs_val_sampled_nt.txt it is also HQ
            df500 = pd.read_csv(os.path.join(metadata_dir, f'val500.csv'))

        with open(subset_val_nt, "rb") as fp:  # Unpickling
            list_val = pickle.load(fp)

        assert df500.shape[0] == df_nt[['couples', 'interacting', 'policy']].merge(df500, on = 'couples').shape[0]
        df500 = df_nt[['couples', 'interacting', 'policy']].merge(df500, on = 'couples')
        df500 = df500[df500.couples.isin(list_val)] # in questo modo ho quasi bilanciato del tutto, ma per avere un bilanciamento al 100% devo fare undersampling
        df500 = undersample_df(df500) #bilanciamento al 100%.
        
    if easy_pretraining:
        pos_multipliers = {25:0.7, 50:0.2, 100:0.1}
        neg_multipliers = {33:0.3, 45:0.1, 55:0.1, 65:0.1,
                           80:0.05, 90:0.05, 100:0.05,
                           120:0.05, 150:0.02, 160:0.02,
                           170:0.02, 180:0.02, 190:0.02,
                           200:0.02, 210:0.02, 220:0.02}

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
            min_n_groups = min_n_groups_val,
            max_n_groups = max_n_groups_val,
        )

    else:

        df500 = df500.sample(frac=1, random_state=23).reset_index(drop = True)
        assert df500.shape[0]>0

        dataset_val = RNADatasetNT500(
            df = df500,
            data_dir = os.path.join(embedding_dir, '32'),
            scaling_factor = scaling_factor,
            min_n_groups = min_n_groups_val,
            max_n_groups = max_n_groups_val,
        )
        policies_val = 'dataset500'

    return dataset_val, policies_val


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

    dataset_train, policies_train = obtain_train_dataset(EASY_PRETRAINING, TRAIN_HQ, FINETUNING, args.min_n_groups_train, args.max_n_groups_train)
    args.policies_train = policies_train

    dataset_val, policies_val = obtain_val_dataset(EASY_PRETRAINING, FINETUNING, args.min_n_groups_val, args.max_n_groups_val)
    args.policies_val = policies_val


     # Save the namespace of the args object to a file using pickle
    with open(os.path.join(args.output_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args.__dict__, f)

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
    for epoch in range(args.start_epoch, args.epochs):
        
        if utils.early_stopping(n_epochs = args.n_epochs_early_stopping, 
                                current_epoch = epoch, 
                                best_model_epoch = utils.best_model_epoch(output_dir / "log.txt")):
            break
            
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

    #nohup python train_binary_cl.py --easy_pretraining &> train_binary_cl_easy.out &
    #nohup python train_binary_cl.py --finetuning &> train_binary_cl_finetuning.out &
    #nohup python train_binary_cl.py --train_hq &> train_binary_cl_hq.out &


    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints', 'binary_cl2')
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')

    EASY_PRETRAINING = args.easy_pretraining
    FINETUNING = args.finetuning
    TRAIN_HQ = args.train_hq    
    SPECIE = args.specie

    if args.modelarch == 1:
        args.use_projection_module = True

    if args.use_projection_module == False:
        args.proj_module_N_channels = 0 # In this way I will not count the parameters of the projection module when I define the n_parameters variable

    seed_everything(123)
    main(args)