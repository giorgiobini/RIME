"""
Credits: mostly copy-pasted from this repo:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import functools
import os
from typing import Optional, List
from collections import defaultdict, deque
import time
import datetime
import pandas as pd
from pathlib import Path
import torch
import torch.distributed as dist
from torch import Tensor
import numpy as np
from .encoding_sequences import prepare_sequence_for_DNABERT

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision

if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__.split(".")[1]) < 7.0:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)

def collate_fn_bert_info_target(batch): 
    
    batch_size = len(batch)
    
    samples = [prepare_rna_branch(batch[i]) for i in range(batch_size)]

    # Calculate the maximum dimensions
    max_dim1 = max(samples[i][0].shape[0] for i in range(batch_size))
    max_dim2 = max(samples[i][1].shape[0] for i in range(batch_size))

    # Initialize rna1 and rna2 tensors with zero-padding
    rna1 = torch.zeros((batch_size, max_dim1), dtype=torch.long)
    rna2 = torch.zeros((batch_size, max_dim2), dtype=torch.long)
    
    # Fill in the tensors with the embeddings
    for i in range(batch_size):
        rna1[i, :samples[i][0].shape[0]] = torch.as_tensor(samples[i][0])
        rna2[i, :samples[i][1].shape[0]] = torch.as_tensor(samples[i][1])
    
    target = [{'interacting': 1 if batch[i].interacting else 0,
              'gene1':batch[i].gene1,
              'gene2':batch[i].gene2,
              'bbox':batch[i].bbox,
              'policy':batch[i].policy,
              'interaction_bbox':batch[i].seed_interaction_bbox,
              'couple_id':batch[i].couple_id}
              for i in range(len(batch))]
    
    batch = ([rna1, rna2], target)
    return batch

def collate_fn_bert(batch): 
    
    ([rna1, rna2], target) = collate_fn_bert_info_target(batch)

    target = torch.tensor([l['interacting'] for l in target])

    batch = ([rna1, rna2], target)

    return batch

def prepare_rna_branch(sample):
    x1, x2, y1, y2 = sample.bbox.coords

    seq1 = sample.gene1_info['cdna'][x1:x2]
    seq2 = sample.gene2_info['cdna'][y1:y2]

    seq1 = torch.tensor(prepare_sequence_for_DNABERT(seq1), dtype=torch.long)
    seq2 = torch.tensor(prepare_sequence_for_DNABERT(seq2), dtype=torch.long)
    return seq1, seq2

def group_averages(arr, k):
    n = arr.shape[0]
    remainder = n % k   # Remainder samples

    # Split the array into groups
    groups = np.split(arr[:n - remainder], k, axis = 0)

    # If there is a remainder, add the remaining samples to the last group
    if remainder > 0:
        groups[-1] = np.concatenate((groups[-1], arr[-remainder:]))
        averages = np.array([np.mean(g, axis = 0) for g in groups])
    else:
        # Calculate the mean along the second axis
        averages = np.mean(groups, axis=1)

    return averages

def prepare_rna_branch_nt(s):
    x1_emb, x2_emb, y1_emb, y2_emb = s.bbox.x1//6, s.bbox.x2//6, s.bbox.y1//6, s.bbox.y2//6

    k1 = (x2_emb-x1_emb)//s.scaling_factor
    k2 = (y2_emb-y1_emb)//s.scaling_factor

    embedding1 = load_embedding(s.embedding1_path)[x1_emb:x2_emb, :]
    embedding2 = load_embedding(s.embedding2_path)[y1_emb:y2_emb, :]

    #embedding1 is (N, 2560)

    rna1 = torch.transpose(
            torch.as_tensor(group_averages(embedding1, k1), dtype=torch.float), 0, 1
            ).unsqueeze(-1)
    rna2 = torch.transpose(
        torch.as_tensor(group_averages(embedding2, k2), dtype=torch.float), 0, 1
        ).unsqueeze(-1)
    return rna1, rna2

@functools.lru_cache(maxsize=1) #10000
def load_embedding(embedding_path):
    return np.load(embedding_path)

def find_extension_from_savepath(savepath):
    # Extracts the extension without the dot
    return os.path.splitext(savepath)[1][1:]

def collate_fn_nt(batch):
    batch_size = len(batch)
    
    # Extract rna1 and rna2 embeddings from the batch

    rna1, rna2 = zip(*[prepare_rna_branch_nt(batch[i]) for i in range(batch_size)])
    
    rna1 = nested_tensor_from_tensor_list(rna1)
    rna2 = nested_tensor_from_tensor_list(rna2)
    
    rna1.tensors = rna1.tensors.squeeze()
    rna2.tensors = rna2.tensors.squeeze()
    
    if batch_size == 1: #add first dimension for batch
        rna1.tensors = rna1.tensors.unsqueeze(0)
        rna2.tensors = rna2.tensors.unsqueeze(0)
    
    #print(rna1.tensors.shape) # torch.Size([b, 2560, max_dim])
    
    # Prepare the target dictionary
    target = [{'interacting': 1 if sample.interacting else 0,
               'gene1': sample.gene1,
               'gene2': sample.gene2,
               'bbox': sample.bbox,
               'policy': sample.policy,
               'interaction_bbox': sample.seed_interaction_bbox,
               'couple_id': sample.couple_id}
              for sample in batch]
    
    target = torch.tensor([l['interacting'] for l in target])

    # Return the batch with rna1, rna2, and the target.
    batch = ([rna1, rna2], target)
    return batch

def prepare_rna_branch_nt2(s, k1, k2):
    x1_emb, x2_emb, y1_emb, y2_emb = s.bbox.x1//6, s.bbox.x2//6, s.bbox.y1//6, s.bbox.y2//6

    # k1 = (x2_emb-x1_emb)//s.scaling_factor
    # k2 = (y2_emb-y1_emb)//s.scaling_factor
    
    # k1, k2 = s.num_groups, s.num_groups
    embedding1 = load_embedding(s.embedding1_path)[x1_emb:x2_emb, :]
    embedding2 = load_embedding(s.embedding2_path)[y1_emb:y2_emb, :]
    return group_averages(embedding1, k1), group_averages(embedding2, k2)

def len1len2(s):
    x1_emb, x2_emb, y1_emb, y2_emb = s.bbox.x1//6, s.bbox.x2//6, s.bbox.y1//6, s.bbox.y2//6
    return x2_emb-x1_emb, y2_emb-y1_emb

def collate_fn_nt2(batch):
    
    batch_size = len(batch)
    
    min_n_groups, max_n_groups = batch[0].min_n_groups, batch[0].max_n_groups
    
    len1, len2 = zip(*[
        len1len2(batch[i]) for i in range(batch_size)
    ])
    
    n_groups1 = min(min(len1), max_n_groups)
    n_groups2 = min(min(len2), max_n_groups)
    
    # second way
    # n_groups1 = int(np.random.randint(min_n_groups, max_n_groups + 1, 1))
    # n_groups2 = int(np.random.randint(min_n_groups, max_n_groups + 1, 1))

    # Extract rna1 and rna2 embeddings from the batch
    embeddings1, embeddings2  = zip(*[
        prepare_rna_branch_nt2(batch[i], n_groups1, n_groups2) for i in range(batch_size)
    ])

    # Calculate the maximum dimensions
    max_dim1 = max(embedding1.shape[0] for embedding1 in embeddings1)
    max_dim2 = max(embedding2.shape[0] for embedding2 in embeddings2)

    # Initialize rna1 and rna2 tensors with zero-padding
    rna1 = torch.zeros((batch_size, max_dim1, 2560), dtype=torch.float32)
    rna2 = torch.zeros((batch_size, max_dim2, 2560), dtype=torch.float32)

    # Fill in the tensors with the embeddings
    for i, embedding1 in enumerate(embeddings1):
        rna1[i, :embedding1.shape[0]] = torch.as_tensor(embedding1)

    for i, embedding2 in enumerate(embeddings2):
        rna2[i, :embedding2.shape[0]] = torch.as_tensor(embedding2)
        
        
    # rna1, rna2 are (batch_size, max_dim, 2560)
    # Transpose rna1 and rna2 to have shape (batch_size, 2560, max_dim)
    rna1 = torch.transpose(rna1, 1, 2)
    rna2 = torch.transpose(rna2, 1, 2)
    
    # Prepare the target dictionary
    target = [{'interacting': 1 if sample.interacting else 0,
               'gene1': sample.gene1,
               'gene2': sample.gene2,
               'bbox': sample.bbox,
               'policy': sample.policy,
               'interaction_bbox': sample.seed_interaction_bbox,
               'couple_id': sample.couple_id}
              for sample in batch]
    
    target = torch.tensor([l['interacting'] for l in target])

    # Return the batch with rna1, rna2, and the target
    batch = ([rna1, rna2], target)
    return batch

def collate_fn_nt3(batch):
    
    batch_size = len(batch)
    
    min_n_groups, max_n_groups = batch[0].min_n_groups, batch[0].max_n_groups
    
    len1, len2 = zip(*[
        len1len2(batch[i]) for i in range(batch_size)
    ])
    
    n_groups1 = min(min(len1), max_n_groups)
    n_groups2 = min(min(len2), max_n_groups)
    
    # second way
    # n_groups1 = int(np.random.randint(min_n_groups, max_n_groups + 1, 1))
    # n_groups2 = int(np.random.randint(min_n_groups, max_n_groups + 1, 1))

    # Extract rna1 and rna2 embeddings from the batch
    embeddings1, embeddings2  = zip(*[
        prepare_rna_branch_nt2(batch[i], n_groups1, n_groups2) for i in range(batch_size)
    ])

    # Calculate the maximum dimensions
    max_dim1 = max(embedding1.shape[0] for embedding1 in embeddings1)
    max_dim2 = max(embedding2.shape[0] for embedding2 in embeddings2)

    # Initialize rna1 and rna2 tensors with zero-padding
    rna1 = torch.zeros((batch_size, max_dim1, 2560), dtype=torch.float32)
    rna2 = torch.zeros((batch_size, max_dim2, 2560), dtype=torch.float32)

    # Fill in the tensors with the embeddings
    for i, embedding1 in enumerate(embeddings1):
        rna1[i, :embedding1.shape[0]] = torch.as_tensor(embedding1)

    for i, embedding2 in enumerate(embeddings2):
        rna2[i, :embedding2.shape[0]] = torch.as_tensor(embedding2)
        
        
    # rna1, rna2 are (batch_size, max_dim, 2560)
    # Transpose rna1 and rna2 to have shape (batch_size, 2560, max_dim)
    rna1 = torch.transpose(rna1, 1, 2)
    rna2 = torch.transpose(rna2, 1, 2)
    
    # Prepare the target dictionary
    target = [{'interacting': 1 if sample.interacting else 0,
               'gene1': sample.gene1,
               'gene2': sample.gene2,
               'bbox': sample.bbox,
               'policy': sample.policy,
               'interaction_bbox': sample.seed_interaction_bbox,
               'couple_id': sample.couple_id}
              for sample in batch]

    # Return the batch with rna1, rna2, and the target
    batch = ([rna1, rna2], target)
    return batch

    
def init_distributed_mode(args):
    #"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print("after readding local var: ", args.gpu, " from args :", args.local_rank )
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    """
    
    #I add 3 new lines
    args.gpu = torch.cuda.device_count()
    args.rank = 0
    args.world_size = args.gpu * 1 #args.world_size = args.gpu * args.nodes
    """

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
    
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

    
    
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

        

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def best_model_epoch(log_path, metric = 'accuracy', maximize = True):
    if os.path.exists(log_path):
        log = pd.read_json(Path(log_path), lines=True)
        if maximize:
            return log.iloc[log['test_{}'.format(metric)].argmax()].epoch
        else:
            return log.iloc[log['test_{}'.format(metric)].argmin()].epoch
    else:
        return 0
    
def save_this_epoch(log_path, metrics = ['accuracy', 'loss'], maximize_list = [True, False], n_top = 8):
    
    if os.path.exists(log_path):
        log = pd.read_json(Path(log_path), lines=True)
    
    save = False

    for i in range(len(metrics)):
        variable = metrics[i]
        maximize = maximize_list[i]

        column = f'test_{variable}'
        current_epoch_value = log.loc[log.epoch.argmax()][column]

        idx = min(n_top-1, log.shape[0]-1)

        if maximize:
            threshold_value = log.sort_values(column, ascending = False).iloc[idx][column]
            if current_epoch_value > threshold_value:
                save = True
                
        elif maximize == False:
            threshold_value = log.sort_values(column, ascending = True).iloc[idx][column]
            if current_epoch_value < threshold_value:
                save = True
                
    return save

def early_stopping(n_epochs, current_epoch, best_model_epoch):
    interrupt = False
    if (current_epoch - best_model_epoch) >= n_epochs:
        interrupt = True
    return interrupt
    
    
def balance_df(df, n_iter = 25):
    toappend = []
    if df[df.ground_truth == 0].shape[0] > df[df.ground_truth == 1].shape[0]:
        for i in range(n_iter):
            negs = df[df.ground_truth == 0]
            poss = df[df.ground_truth == 1]
            toappend.append(pd.concat([negs.sample(len(poss)), poss], axis = 0))
    else:
        for i in range(n_iter):
            negs = df[df.ground_truth == 0]
            poss = df[df.ground_truth == 1]
            toappend.append(pd.concat([poss.sample(len(negs)), negs], axis = 0))
    balanced = pd.concat(toappend, axis = 0)
    return balanced


def undersample_df(df, column='ground_truth'):
    # Separate the DataFrame into the two classes
    class_0 = df[df[column] == 0]
    class_1 = df[df[column] == 1]

    if len(class_0) > len(class_1):
        majority_class = class_0
        minority_class = class_1
    else:
        majority_class = class_1
        minority_class = class_0

    num_minority_samples = len(minority_class)
    majority_class_undersampled = majority_class.sample(num_minority_samples)
    df_balanced = pd.concat([majority_class_undersampled, minority_class])
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

    return df_balanced

def is_unbalanced(subset, column='ground_truth'):
    class_counts = subset[column].value_counts()
    return abs(class_counts[0] - class_counts[1]) > 0

def obtain_majority_minority_class(subset):
    class_0 = subset[subset.ground_truth == 0]
    class_1 = subset[subset.ground_truth == 1]
    if len(class_1)>len(class_0):
        majority_class = class_1
        minority_class = class_0
    else:
        majority_class = class_0
        minority_class = class_1
    return majority_class, minority_class