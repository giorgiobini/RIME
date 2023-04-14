"""
Credits: mostly copy-pasted from this repo:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

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
    
def collate_fn(batch): 
    
    batch_size = len(batch)
    
    # batch is a list of dicts of lenght == batch_size
    
    samples = [prepare_rna_branch(batch[i]) for i in range(batch_size)]
    
    target = [{'interacting': 1 if batch[i].interacting else 0,
              'gene1':batch[i].gene1,
              'gene2':batch[i].gene2,
              'bbox':batch[i].bbox,
              'policy':batch[i].policy,
              'interaction_bbox':batch[i].seed_interaction_bbox,
              'couple_id':batch[i].couple_id}
              for i in range(len(batch))]
    
    rna1_ss, rna1, rna2_ss, rna2 = list(map(list, zip(*samples)))
    
    #print(rna1_ss[0].shape) #torch.Size([3, N, 1])
    #print(rna1[0].shape) # torch.Size([1, N-3, 1])
    
    rna1 = nested_tensor_from_tensor_list(rna1)
    rna2 = nested_tensor_from_tensor_list(rna2)
    
    #print(rna1.mask.shape) #torch.Size([b, max(N-3), 1])
    #print(rna1.tensors.shape) #torch.Size([b, 1, max(N-3), 1])
    
    rna1.tensors = rna1.tensors.squeeze()
    rna2.tensors = rna2.tensors.squeeze()
    
    
    #print(rna1.tensors.shape) #torch.Size([b, max(N-3)])
    
    rna1_ss = nested_tensor_from_tensor_list(rna1_ss).tensors.squeeze()
    rna2_ss = nested_tensor_from_tensor_list(rna2_ss).tensors.squeeze()
    
    if batch_size == 1: #add first dimension for batch
        rna1.tensors = rna1.tensors.unsqueeze(0)
        rna2.tensors = rna2.tensors.unsqueeze(0)
        rna1_ss = rna1_ss.unsqueeze(0)
        rna2_ss = rna2_ss.unsqueeze(0)
    
    #print(rna1_ss.shape) # torch.Size([b, 3, max(N)])
    
    rna1 = [rna1_ss, rna1]
    rna2 = [rna2_ss, rna2]
    
    batch = ([rna1, rna2], target)
    return batch


def prepare_rna_branch(sample):
    x1, x2, y1, y2 = sample.bbox.coords

    dot_br1 =  sample.gene1_info['dot_br'][x1:x2]
    seq1 = sample.gene1_info['cdna'][x1:x2]

    dot_br1 = dot2onehot(dot_br1)

    dot_br2 = sample.gene2_info['dot_br'][y1:y2]
    seq2 = sample.gene2_info['cdna'][y1:y2]

    dot_br2 = dot2onehot(dot_br2)
    
    seq1 = torch.tensor(prepare_sequence_for_DNABERT(seq1), dtype=torch.long)
    seq2 = torch.tensor(prepare_sequence_for_DNABERT(seq2), dtype=torch.long)

    #I need to unsqueeze and permute for the nested_tensor_from_tensor_list function
    return dot_br1.permute(1, 0).unsqueeze(-1), seq1.unsqueeze(-1).unsqueeze(0), dot_br2.permute(1, 0).unsqueeze(-1), seq2.unsqueeze(-1).unsqueeze(0)
    
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
    

def early_stopping(n_epochs, current_epoch, best_model_epoch):
    interrupt = False
    if (current_epoch - best_model_epoch) >= n_epochs:
        interrupt = True
    return interrupt
    