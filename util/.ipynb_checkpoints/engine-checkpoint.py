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

@torch.no_grad()
def random_classifier_stats(model, criterion, postprocessors, data_loader, device, output_dir, frequency_print = 1000):
    # I will give to the ranodm classifier the median width and height of the bounding boxes in the training set.
    # I will keep the same pred_logits (so that we can compare the cardinality error with our model) and I will create random bounding boxes.
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Random:'
    for samples, targets in metric_logger.log_every(data_loader, frequency_print, header):
        fake_pred_boxes= []
        rna1, rna2 = samples
        rna1 = rna1.to(device)
        rna2 = rna2.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(rna1, rna2)
        for target in targets: 
            real_box = target['boxes'][0]
            xc, yc, w, h = real_box
            w, h = torch.tensor(0.11, dtype=torch.float32, device=w.device), torch.tensor(0.11, dtype=torch.float32, device=w.device) #median wdith, height of the training bounding boxes
            fake_preds = []
            for i in range(20):
                xc_fake = (np.random.randint(0, 101) - (w/2)*100)/100 #xc_fake = max(0, np.random.rand() - w/2)
                yc_fake = (np.random.randint(0, 101) - (w/2)*100)/100 #yc_fake = max(0, np.random.rand() - h/2)
                fake_preds.append([xc_fake,yc_fake,w,h])
            fake_pred_boxes.append(fake_preds)
        
        fake_pred_boxes = torch.as_tensor(np.array(fake_pred_boxes))
        fake_outputs = {'pred_logits':outputs['pred_logits'],
                       'pred_boxes':fake_pred_boxes}
        
        loss_dict = criterion(fake_outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](fake_outputs, orig_target_sizes)

        res = {target['pairs_id'].item(): output for target, output in zip(targets, results)}
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
 
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, output_dir, frequency_print = 1000):
    #model.projection_module.eval()
    #model.detr.transformer.encoder.eval() #this was the only one not commented during the training...
    #model.detr.transformer.eval()
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, frequency_print, header):
        rna1, rna2 = samples
        rna1 = rna1.to(device)
        rna2 = rna2.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print(len(targets)) #batch_size
        #print(targets)
        
        outputs = model(rna1, rna2)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['pairs_id'].item(): output for target, output in zip(targets, results)}

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
 
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, frequency_print: int = 1000): #
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for samples, targets in metric_logger.log_every(data_loader, frequency_print, header):
        rna1, rna2 = samples
        rna1 = rna1.to(device)
        rna2 = rna2.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(rna1, rna2)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
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
        rna1[0] = rna1[0].to(device)
        rna2[0] = rna2[0].to(device)
        rna1[1].tensors = rna1[1].tensors.to(device)
        rna2[1].tensors = rna2[1].tensors.to(device)
        rna1[1].mask = rna1[1].mask.to(device)
        rna2[1].mask = rna2[1].mask.to(device)
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
        rna1[0] = rna1[0].to(device)
        rna2[0] = rna2[0].to(device)
        rna1[1].tensors = rna1[1].tensors.to(device)
        rna2[1].tensors = rna2[1].tensors.to(device)
        rna1[1].mask = rna1[1].mask.to(device)
        rna2[1].mask = rna2[1].mask.to(device)
        
        outputs = model(rna1, rna2)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        metric_logger.update(**loss_dict)        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
 
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats