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
from util.engine import train_one_epoch_mlp as train_one_epoch
from util.engine import evaluate_mlp as evaluate
from models.mlp import build as build_model 
import util.misc as utils
import json
from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import os
import sys
sys.path.insert(0, '..')
from config import *

class MeanEmbDataset(Dataset):
    def __init__(self, folder_path, augment):
        self.folder_path = folder_path
        self.augment = augment
        self.samples = []
        self.labels = []
        for class_label in ['class_0', 'class_1']:
            class_folder = os.path.join(folder_path, class_label)
            for sample_file in os.listdir(class_folder):
                sample = np.load(os.path.join(class_folder, sample_file))
                self.samples.append(sample)
                self.labels.append(int(class_label.split('_')[1]))

    def __len__(self):
        return len(self.samples)

    def shuffle_vectors(self, sample):
        # data augmentation
        # Split the concatenated sample into two vectors of size (2560,) 
        vec1, vec2 = np.split(sample, 2)
        # Shuffle the order of the two vectors randomly
        if np.random.rand() < 0.5:
            return np.concatenate([vec1, vec2])
        else:
            return np.concatenate([vec2, vec1])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.augment:
            sample = self.shuffle_vectors(sample)  # Shuffle the order of the two feature vectors
        label = self.labels[idx]
        return torch.from_numpy(sample).float(), label

def create_data_loader_training(folder_path, batch_size, num_workers=0):
    dataset = MeanEmbDataset(folder_path, True)

    # count the number of samples for each class
    class_sample_count = [0, 0]
    for label in dataset.labels:
        class_sample_count[label] += 1
    
    num_samples = sum(class_sample_count)
    class_weights = [num_samples/class_sample_count[i] for i in range(len(class_sample_count))]
    weights = [class_weights[dataset.labels[i]] for i in range(int(num_samples))]

    # create a weighted sampler to sample the data
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    # create a data loader with the weighted sampler
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers = num_workers)
    return data_loader

def create_data_loader_test(folder_path, batch_size, shuffle=False, num_workers=0):
    dataset = MeanEmbDataset(folder_path, False)
    
    # create a data loader with the default sampler
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers)
    return data_loader


def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=6000, type=int)

    # * Model
    parser.add_argument('--num_hidden_layers', default=2, type=int,
                        help="Number of hidden layers in the MLP")
    parser.add_argument('--dividing_factor', default=100, type=int,
                        help="If the input is 5120, the first layer of the MLP is 5120/dividing_factor")
    parser.add_argument('--dropout_prob', default=0.1, type=float,
                        help="Dropout applied in the model")

    # dataset parameters
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing') # originally was 'cuda'
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--embedding_layer', default='22', 
                        help='Which is the embedding layer you cutted the NT model')
    parser.add_argument('--k', default='999', 
                    help='k is the k-group based parameters. 999 means you take the mean of the whole embedding sequence.')
    
    # training parameters
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--n_epochs_early_stopping', default=3000)
    return parser

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def main(args):
    seed_worker(4567)

    #Se il data loader non funziona leva il commento
    #torch.multiprocessing.set_sharing_strategy('file_system')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_dir = Path(args.output_dir)

    if os.path.isfile(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    
    folder_training = os.path.join(args.layer_folder, 'training')
    folder_val = os.path.join(args.layer_folder, 'val')
    data_loader_train = create_data_loader_training(folder_training, args.batch_size, num_workers=args.num_workers)
    data_loader_val = create_data_loader_test(folder_val, args.batch_size, True, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        if utils.early_stopping(n_epochs = args.n_epochs_early_stopping, 
                                current_epoch = epoch, 
                                best_model_epoch = utils.best_model_epoch(output_dir / "log.txt")):
            break
            
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch)
    
        test_stats = evaluate(model, criterion, data_loader_val, device)
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        best_model_epoch = utils.best_model_epoch(output_dir / "log.txt")
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                
            if best_model_epoch == epoch:
                checkpoint_paths.append(output_dir / f'best_model.pth')
                
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
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
    
    layer = str(args.embedding_layer)
    args.layer_folder = os.path.join(embedding_dir, args.k, layer)
    folder_name = f'NTlayer{layer}_dividing_factor{args.dividing_factor}_hiddenlayers{args.num_hidden_layers}'
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints', 'nt', folder_name)
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')
    
    main(args)
