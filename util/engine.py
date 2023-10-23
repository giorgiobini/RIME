# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import datetime
import time
import sys
from typing import Iterable
import torch
from . import misc as utils
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                              data_loader: Iterable, optimizer: torch.optim.Optimizer,
                              device: torch.device, epoch: int, grad_accumulate:int = 1):

    #if you want batch_size = 32, but you can load only 2 in gpu, then grad_accumulate = 16
    model.train()
    criterion.train()
    
    real = []
    pred = []
    losses = []

    start_time = time.time()

    for k, (samples, labels) in enumerate(data_loader):
        rna1, rna2 = samples
        rna1 = rna1.to(device)
        rna2 = rna2.to(device)

        outputs = model(rna1, rna2)
        
        del rna1, rna2
        labels = labels.to(device)
        loss = criterion(outputs, labels) #loss = torch.nn.functional.cross_entropy

        loss.backward() #step di accumulazione

        if k%grad_accumulate == 0:
            #step completo
            optimizer.step()
            optimizer.zero_grad()
        
        losses.append(loss.detach().cpu().item())
        pred.append(outputs.detach().cpu()) 
        real.append(labels.detach().cpu())

    if k%grad_accumulate!=0:
        optimizer.step()
        optimizer.zero_grad()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
    stats = calc_metrics(torch.cat(pred), torch.cat(real))
    stats['loss'] = torch.mean(torch.tensor(losses)).item()
    print('Epoch: [{}], (Time: {})'.format(epoch, total_time_str))
    print("Averaged stats:", stats)
    return stats

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    real = []
    pred = []
    losses = []

    start_time = time.time()

    for samples, labels in data_loader:
        rna1, rna2 = samples
        rna1 = rna1.to(device)
        rna2 = rna2.to(device)
        
        outputs = model(rna1, rna2)
        del rna1, rna2
        labels = labels.to(device)
        loss = criterion(outputs, labels)
        
        losses.append(loss.detach().cpu().item())
        pred.append(outputs.detach().cpu()) 
        real.append(labels.detach().cpu())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
 
    stats = calc_metrics(torch.cat(pred), torch.cat(real))
    stats['loss'] = float(torch.mean(torch.tensor(losses)))
    print('Test (Time: {})'.format(total_time_str))
    print("Averaged stats:", stats)
    return stats


@torch.no_grad()
def calc_metrics(predictions, ground_truth, beta = 2):
    """ 
    Compute accuracy, precision, recall, f1
    
    add true positive rate, ec..
    """

    pred = (torch.argmax(predictions, dim=1) == 1)
    label = (ground_truth == 1)
    TP = (pred & label).sum().float()
    TN = ((~pred) & (~label)).sum().float()
    FP = (pred & (~label)).sum().float()
    FN = ((~pred) & label).sum().float()
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    F2 = F2.mean(0)
    TNR = torch.mean(TN / (TN + FP + 1e-12))
    NPV = torch.mean(TN / (TN + FN + 1e-12))

    metrics = {'accuracy': accuracy.item(), 'precision': precision.item(), 'recall':recall.item(), 'F2': F2.item(), 'specificity': TNR.item(), 'NPV': NPV.item()} 
    return metrics