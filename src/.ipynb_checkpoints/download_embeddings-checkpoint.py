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

class MeanEmbDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
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

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        return torch.from_numpy(sample).float(), label


def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    # * Model
    parser.add_argument('--num_hidden_layers', default=1, type=int,
                        help="Number of hidden layers in the MLP")
    parser.add_argument('--dropout_prob', default=0.2, type=float,
                        help="Dropout applied in the model")

    # dataset parameters
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing') # originally was 'cuda'
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--embedding_layer', default='', 
                        help='Which is the embedding layer you cutted the NT model')
    
    # training parameters
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--n_epochs_early_stopping', default=10)
    return parser

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def main(args):
    seed_worker(123)

    #Se il data loader non funziona leva il commento
    #torch.multiprocessing.set_sharing_strategy('file_system')

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_dir = Path(args.output_dir)

    if os.path.isfile(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    
    dataset_train = MeanEmbDataset(os.path.join(args.layer_folder, 'training'))
    dataset_val = MeanEmbDataset(os.path.join(args.layer_folder, 'val'))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_train = DataLoader(dataset_train,  args.batch_size, sampler=sampler_train,  drop_last=False, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)
        
    device = torch.device(args.device)
    model = build_model(args)
    model.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
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
    
        test_stats = evaluate(model, criterion, data_loader_val)
            
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
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                
            if best_model_epoch == epoch:
                checkpoint_paths.append(output_dir / f'best_model.pth')
                
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
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
    #nohup python train_binary_cl.py --drop_secondary_structure=true &> train_binary_cl.out &

    ROOT_DIR = os.path.dirname(os.path.abspath('.'))
    original_files_dir = os.path.join(ROOT_DIR, 'dataset', 'original_files')
    processed_files_dir = os.path.join(ROOT_DIR, 'dataset', 'processed_files')
    rna_rna_files_dir = os.path.join(ROOT_DIR, 'dataset', 'rna_rna_pairs')
    nt_data_dir = os.path.join(processed_files_dir, "nt_data")
    embedding_dir = os.path.join(nt_data_dir, "embeddings")

    parser = argparse.ArgumentParser('Training', parents=[get_args_parser()])
    args = parser.parse_args()
    
    layer = str(args.embedding_layer)
    args.layer_folder = os.path.join(embedding_dir, layer)
    args.input_size = 2560 #input size depends on the layer you cut? or the embedding dim is fixed?
    folder_name = layer + '_' + str(args.num_hidden_layers)
    args.output_dir = os.path.join(ROOT_DIR, 'checkpoints', 'nt', folder_name)
    args.dataset_path = os.path.join(ROOT_DIR, 'dataset')
    
    main(args)
