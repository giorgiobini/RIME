# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
from . import misc as utils
import numpy as np


def train_one_epoch_mlp(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, frequency_print: int = 500): #frequency_print: int = 1000
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for inputs, labels in metric_logger.log_every(data_loader, frequency_print, header):
        rna1, rna2 = inputs
        rna1=rna1.to(device)
        rna2=rna2.to(device)
        # Forward pass
        outputs = model(rna1, rna2)
        labels = torch.tensor([l['interacting'] for l in labels]).to(device)
        loss = criterion(outputs, labels) #loss = torch.nn.functional.cross_entropy

        #se sono in uno step di accumalazione, fai solo loss.backword(). Se sono in uno step finale, fai tutti e 3. Ma l'ordine deve essere loss.backward(), optimizer.step(), optimizer.zero_grad()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metric_dict = calc_metrics(outputs, labels, beta = 2)
        metric_dict['loss'] = loss
        
        metric_logger.update(**metric_dict)
        
    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_mlp(model, criterion, data_loader, device, frequency_print = 1000):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for inputs, labels in metric_logger.log_every(data_loader, frequency_print, header):
        # Forward pass
        rna1, rna2 = inputs
        rna1=rna1.to(device)
        rna2=rna2.to(device)
        outputs = model(rna1, rna2)
        
        labels = torch.tensor([l['interacting'] for l in labels]).to(device)
        loss = criterion(outputs, labels)
        
        metric_dict = calc_metrics(outputs, labels, beta = 2)
        metric_dict['loss'] = loss
        
        metric_logger.update(**metric_dict)       

    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_binary_cl(model: torch.nn.Module, criterion: torch.nn.Module,
                              data_loader: Iterable, optimizer: torch.optim.Optimizer,
                              device: torch.device, epoch: int, max_norm: float = 0, frequency_print: int = 500): #frequency_print: int = 1000
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for samples, targets in metric_logger.log_every(data_loader, frequency_print, header):
        rna1, rna2 = samples
        rna1.tensors = rna1.tensors.to(device)
        rna2.tensors = rna2.tensors.to(device)
        rna1.mask = rna1.mask.to(device)
        rna2.mask = rna2.mask.to(device)
        outputs = model(rna1, rna2)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        loss_value = losses.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(losses)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_binary_cl(model, criterion, postprocessors, data_loader, device, output_dir, frequency_print = 1000):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, frequency_print, header):
        rna1, rna2 = samples
        rna1.tensors = rna1.tensors.to(device)
        rna2.tensors = rna2.tensors.to(device)
        rna1.mask = rna1.mask.to(device)
        rna2.mask = rna2.mask.to(device)
        
        outputs = model(rna1, rna2)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        metric_logger.update(**loss_dict)        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
 
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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

    metrics = {'accuracy': accuracy, 'precision': precision, 'recall':recall, 'F2': F2, 'specificity': TNR, 'NPV': NPV} 
    return metrics