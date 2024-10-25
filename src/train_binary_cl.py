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
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from models.nt_classifier import build as build_model 
import util.misc as utils
from util.params_dataloader import load_windows_and_multipliers
import json
from dataset.data import (
    RNADataset,
    RNADatasetNT,
    RNADatasetNT500,
    EasyPosAugment,
    EasyNegAugment,
    HardNegAugment,
    InteractionSelectionPolicy,
    SmartNegAugment,
    seed_everything,
    clean_nt_dataframes_before_class_input,
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
    
class MergedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        for i, length in enumerate(self.lengths):
            if index < length:
                return self.datasets[i][index]
            index -= length

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
    
    # hyperparameters
    parser.add_argument('--class_1_weight', default=1.0, type=float, help="The higher is the lower will be the false positives. For instance, a value of 2.5 (penalizes class 1 errors 2.5 times more than class 0 errors).")
    parser.add_argument('--lr', default=5e-5, type=float) #5e-5 con batch_size = 32, 1e-4 con batch_size = 8
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--grad_accumulate', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Training mood
    parser.add_argument('--train_dataset', default='paris',
                        help='can be paris, mario, ricseq, splash')
    parser.add_argument('--val_dataset', default='paris',
                        help='can be paris, mario, ricseq, splash')
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
                        help="Can be 1, 2. Architecture 1 has convolution projection on each branch and then contact matrix. Architecture 2 has mlp concatenation for each token combination and the contact matrix is built from this concat.")
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
    parser.add_argument('--per_sample_p', default=0.5, type=float, help='for each epoch, I can see all the postives (per_sample_p = 1), half of them (per_sample_p = 0.5) or a fraction of them (e.g. per_sample_p = 0.25). The number of negatives is proportional also to this value.')
    parser.add_argument('--proportion_sn', default=0.5, type=float, help='when training for interactors task, I will keep a proportion of SN, of a proportion of HN and a proportion of EN')
    parser.add_argument('--proportion_hn', default=0.3, type=float, help='when training for interactors task, I will keep a proportion of SN, of a proportion of HN and a proportion of EN')
    parser.add_argument('--proportion_en', default=0.2, type=float, help='when training for interactors task, I will keep a proportion of SN, of a proportion of HN and a proportion of EN')
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
    parser.add_argument('--desired_length', default=200, type=int,
                       help='The model will be trained and tested in RNAs with an approximal length of desired_length')


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
    parser.add_argument('--n_epochs_early_stopping', default=200)
    return parser

def obtain_train_dataset_paris(dimension, train_hq, finetuning, per_sample_p, proportion_sn, proportion_hn, proportion_en, min_n_groups_train, max_n_groups_train, specie, scaling_factor = 5):

    if train_hq:
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_HQ.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_HQ.csv'))
        if finetuning:
            subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_train_val_fine_tuning_nt_HQ.txt")
        else:
            subset_train_nt = os.path.join(rna_rna_files_dir, 'gene_pairs_training_nt_HQ.txt')
    else:
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt.csv'))
        df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt.csv'))
        if finetuning:
            subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_train_val_fine_tuning_nt.txt")
        else:
            subset_train_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_training_nt.txt")
    df_nt, df_genes_nt = clean_nt_dataframes_before_class_input(df_nt, df_genes_nt)

    with open(subset_train_nt, "rb") as fp:  # Unpickling
        list_train = pickle.load(fp)
        
    if specie in ['human', 'mouse']:
        paris = pd.read_csv(os.path.join(processed_files_dir, f'paris.csv'))
        couples_to_keep = set(paris[paris.specie == specie].couples)
        df_nt = df_nt[df_nt.couples_id.isin(couples_to_keep)].reset_index(drop = True)

    vc_train = df_nt[df_nt.couples.isin(list_train)].interacting.value_counts()
    assert vc_train[False]>vc_train[True]
    unbalance_factor = 1 - (vc_train[False] - vc_train[True]) / vc_train[False]


    pos_multipliers, neg_windows = load_windows_and_multipliers(dimension, 'paris', specie, train_hq, finetuning)
    neg_multipliers = pos_multipliers

    sn_per_sample, hn_per_sample, en_per_sample = get_per_sample_from_proportion(per_sample_p, unbalance_factor, proportion_sn, proportion_hn, proportion_en)
    policies_train = obtain_policies_object(per_sample_p, sn_per_sample, hn_per_sample, en_per_sample, pos_multipliers, neg_multipliers, neg_windows)

    dataset_train  = obtain_dataset_object(policies_train, df_genes_nt, df_nt, subset_train_nt, scaling_factor, min_n_groups_train, max_n_groups_train)
    
    return dataset_train, policies_train

def get_per_sample_from_proportion(per_sample_p, unbalance_factor, proportion_sn, proportion_hn, proportion_en):
    sn_per_sample = unbalance_factor * (per_sample_p) * proportion_sn
    hn_per_sample = (per_sample_p) * proportion_hn
    en_per_sample = unbalance_factor * (per_sample_p) * proportion_en
    return sn_per_sample, hn_per_sample, en_per_sample


def obtain_policies_object(per_sample_p, sn_per_sample, hn_per_sample, en_per_sample, pos_multipliers, neg_multipliers, neg_windows):
    policies_train = [
        EasyPosAugment(
            per_sample=per_sample_p,
            interaction_selection=InteractionSelectionPolicy.LARGEST,
            width_multipliers=pos_multipliers,
            height_multipliers=pos_multipliers,
        ),
    ]

    if sn_per_sample>0:
        policies_train.append(
            SmartNegAugment(
                per_sample=sn_per_sample,
                interaction_selection=InteractionSelectionPolicy.LARGEST,
                width_multipliers=neg_multipliers,
                height_multipliers=neg_multipliers,
            ),
        )
    if hn_per_sample>0:
        policies_train.append(
            HardNegAugment(
                per_sample=hn_per_sample,
                width_windows=neg_windows,
                height_windows=neg_windows,
            ),
        )
    if en_per_sample>0:
        policies_train.append(
            EasyNegAugment(
                per_sample=en_per_sample,
                width_windows=neg_windows,
                height_windows=neg_windows,
            ),
        )
    return policies_train

def obtain_dataset_object(policies, df_genes_nt, df_nt, subset_nt, scaling_factor, min_n_groups, max_n_groups):
    dataset = RNADatasetNT(
            gene2info=df_genes_nt,
            interactions=df_nt,
            subset_file=subset_nt,
            augment_policies=policies,
            data_dir = os.path.join(embedding_dir, '32'),
            scaling_factor = scaling_factor,
            min_n_groups = min_n_groups,
            max_n_groups = max_n_groups,
    )
    return dataset

def obtain_val_dataset(dataset, dimension, finetuning, min_n_groups_val, max_n_groups_val, specie, scaling_factor = 5, filter_test = True):
    if dataset == 'paris':
        dataset_val, policies_val = obtain_val_dataset_paris(dimension, finetuning, min_n_groups_val, max_n_groups_val, specie, scaling_factor)
    else:
        raise NotImplementedError #I have to correct everything below. Remember that now we have 'df_nt_id' not anymore 'couples' on df500.
        assert False
        assert dataset in ['mario', 'ricseq', 'splash']
        df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_{dataset}.csv'))
        data_dir = os.path.join(rna_rna_files_dir, f'{dataset}')

        file_test = os.path.join(data_dir, 'gene_pairs_test.txt')
        with open(file_test, "rb") as fp:   # Unpickling
            test_couples = pickle.load(fp)

        df500 = pd.read_csv(os.path.join(metadata_dir, f'{dataset}{dimension}.csv'))
        assert df500.shape[0] == df_nt[['couples', 'couples_id']].merge(df500, on = 'couples').shape[0]
        
        
        df500['interacting'] = False
        df500.loc[df500['policy'] == 'easypos', 'interacting'] = True
        
        df500 = df_nt[['couples', 'couples_id']].merge(df500, on = 'couples')
        
        if filter_test:
            df500 = df500[df500.couples_id.isin(test_couples)]
            
        df500 = df500[df500.policy.isin(['easypos', 'smartneg'])]
        df500 = undersample_df(df500)

        df500 = df500.sample(frac=1, random_state=23).reset_index(drop = True)
        assert df500.shape[0]>0

        dataset_val = RNADatasetNT500(
            df = df500,
            data_dir = os.path.join(embedding_dir, '32'),
            scaling_factor = scaling_factor,
            min_n_groups = min_n_groups_val,
            max_n_groups = max_n_groups_val,
        )
    return dataset_val, 'dataset500'


def obtain_val_dataset_paris(dimension, finetuning, min_n_groups_val, max_n_groups_val, specie, scaling_factor = 5):
    
    if finetuning:
        subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_test_sampled_nt_HQ.txt") # gene_pairs_test_sampled_nt.txt it is also HQ
        df500 = pd.read_csv(os.path.join(metadata_dir, f'test_HQ{dimension}.csv'))
    else:
        subset_val_nt = os.path.join(rna_rna_files_dir, f"gene_pairs_val_sampled_nt_HQ.txt") # gene_pairs_val_sampled_nt.txt it is also HQ
        df500 = pd.read_csv(os.path.join(metadata_dir, f'val_HQ{dimension}.csv'))

    with open(subset_val_nt, "rb") as fp:  # Unpickling
        list_val = pickle.load(fp)

    if specie in ['human', 'mouse']:
        paris = pd.read_csv(os.path.join(processed_files_dir, f'paris.csv'))
        couples_to_keep = set(paris[paris.specie == SPECIE].couples)
        df500 = df500[(df500.g1 + '_' + df500.g2).isin(couples_to_keep)].reset_index()
    

    df500['interacting'] = False
    df500.loc[df500['policy'] == 'easypos', 'interacting'] = True

    df500 = df500[df500.df_nt_id.isin(list_val)] # in questo modo ho quasi bilanciato del tutto, ma per avere un bilanciamento al 100% devo fare undersampling
    df500 = undersample_df(df500) #bilanciamento al 100%.

    df500 = df500.sample(frac=1, random_state=23).reset_index(drop = True)
    assert df500.shape[0]>0

    dataset_val = RNADatasetNT500(
        df = df500,
        data_dir = os.path.join(embedding_dir, '32'),
        scaling_factor = scaling_factor,
        min_n_groups = min_n_groups_val,
        max_n_groups = max_n_groups_val,
    )

    return dataset_val, 'dataset500'
    

def obtain_train_splash_mario_ricseq_dataset(dataset, dimension, per_sample_p, proportion_sn, proportion_hn, proportion_en, min_n_groups_train, max_n_groups_train, scaling_factor = 5):
    assert dataset in ['mario', 'ricseq', 'splash']
    df_nt = pd.read_csv(os.path.join(metadata_dir, f'df_nt_{dataset}.csv'))
    df_genes_nt = pd.read_csv(os.path.join(metadata_dir, f'df_genes_nt_{dataset}.csv'))
    df_nt, df_genes_nt = clean_nt_dataframes_before_class_input(df_nt, df_genes_nt)
    data_dir = os.path.join(rna_rna_files_dir, f'{dataset}')

    pos_multipliers, neg_windows = load_windows_and_multipliers(dimension, 'splash', np.nan, np.nan, np.nan)
    neg_multipliers = pos_multipliers

    vc_train = df_nt.interacting.value_counts()
    unbalance_factor = 1 - (vc_train[False] - vc_train[True]) / vc_train[False]
    
    data_dir = os.path.join(rna_rna_files_dir, f'{dataset}')
    file_training = os.path.join(data_dir, 'gene_pairs_training.txt')
    with open(file_training, "rb") as fp:   # Unpickling
        train_couples = pickle.load(fp)

    train_nt = df_nt[df_nt.couples_id.isin(train_couples)]

    sn_per_sample, hn_per_sample, en_per_sample = get_per_sample_from_proportion(per_sample_p, unbalance_factor, proportion_sn, proportion_hn, proportion_en)

    policies_train = obtain_policies_object(per_sample_p, sn_per_sample, hn_per_sample, en_per_sample, pos_multipliers, neg_multipliers, neg_windows)

    dataset_train  = obtain_dataset_object(policies_train, df_genes_nt, df_nt, '', scaling_factor, min_n_groups_train, max_n_groups_train)
    return dataset_train, policies_train

def obtain_train_dataset(dataset, dimension, train_hq, finetuning, per_sample_p, proportion_sn, proportion_hn, proportion_en, min_n_groups_train, max_n_groups_train, specie, scaling_factor = 5):
    if dataset == 'paris':
        dataset_train, policies_train = obtain_train_dataset_paris(dimension, train_hq, finetuning, per_sample_p, proportion_sn, proportion_hn, proportion_en, min_n_groups_train, max_n_groups_train, specie, scaling_factor)
    elif dataset == 'psoralen':
        dataset_paris, policies_train = obtain_train_dataset_paris(dimension, train_hq, finetuning, per_sample_p, proportion_sn, proportion_hn, proportion_en, min_n_groups_train, max_n_groups_train, specie, scaling_factor)
        dataset_splash, _ = obtain_train_splash_mario_ricseq_dataset('splash', dimension, per_sample_p, proportion_sn, proportion_hn, proportion_en, min_n_groups_train, max_n_groups_train, scaling_factor)
        dataset_train = MergedDataset([dataset_splash, dataset_paris])
    else:
        dataset_train, policies_train = obtain_train_splash_mario_ricseq_dataset(dataset, dimension, per_sample_p, proportion_sn, proportion_hn, proportion_en, min_n_groups_train, max_n_groups_train, scaling_factor)
    return dataset_train, policies_train


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

    dataset_train, policies_train = obtain_train_dataset(TRAIN_DATASET, DIMENSION, TRAIN_HQ, FINETUNING, args.per_sample_p, args.proportion_sn, args.proportion_hn, args.proportion_en, args.min_n_groups_train, args.max_n_groups_train, SPECIE)
    args.policies_train = policies_train

    dataset_val, policies_val = obtain_val_dataset(VAL_DATASET, DIMENSION, FINETUNING, args.min_n_groups_val, args.max_n_groups_val, SPECIE)
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

    weights = torch.tensor([args.class_1_weight, 1.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
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
            
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, grad_accumulate = args.grad_accumulate)
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
               
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        
        if args.class_1_weight == 1:
            save_this_epoch = utils.save_this_epoch(os.path.join(output_dir, "log.txt"), metrics = ['accuracy', 'F2', 'loss'], maximize_list = [True, True, False], n_top = 15)

        elif args.class_1_weight > 1:
            save_this_epoch = utils.save_this_epoch(os.path.join(output_dir, "log.txt"), metrics = ['F2', 'precision', 'specificity', 'loss'], maximize_list = [True, True, True, False], n_top = 15)
        else:
            raise NotImplementedError

        if save_this_epoch:
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
            
        #create the new dataset
        dataset_train, policies_train = obtain_train_dataset(TRAIN_DATASET, DIMENSION, TRAIN_HQ, FINETUNING, args.per_sample_p, args.proportion_sn, args.proportion_hn, args.proportion_en, args.min_n_groups_train, args.max_n_groups_train, SPECIE)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_nt2, num_workers=args.num_workers)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn_nt2, num_workers=args.num_workers)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    
    
if __name__ == '__main__':
    #run me with: -> 

    #nohup python train_binary_cl.py &> train_binary_cl.out &
    
    #nohup python train_binary_cl.py --class_1_weight=1.5 &> train_binary_cl.out &
    #nohup python train_binary_cl.py --class_1_weight=1.5 --finetuning &> train_binary_cl_finetuning.out &

    #nohup python train_binary_cl.py --finetuning &> train_binary_cl_finetuning.out &

    #nohup python train_binary_cl.py --train_dataset=psoralen &> train_binary_cl_psoralen.out &
    #nohup python train_binary_cl.py --train_dataset=psoralen --finetuning &> train_binary_cl_finetuning_psoralen.out &
    
    #nohup python train_binary_cl.py --train_hq &> train_binary_cl_hq.out & # TODO train_hq

    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints', 'binary_cl2')
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')

    DIMENSION = args.desired_length
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