from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
        

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, 
                 num_channels: int, interediate_resnet_layer:int, 
                 n_channels_backbone_out:int, last_layer_intermediate_channels:int):
        super().__init__()
        
        #for name, parameter in backbone.named_parameters():
        #    if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #        parameter.requires_grad_(False)
        
        assert interediate_resnet_layer in [2,3,4]
        return_layers = {'layer{}'.format(interediate_resnet_layer): "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    
def reduce_resnet18(backbone, out_dims, interediate_resnet_layer, mini_resnet_channels = 128):
    
        original_dims = 64

        
        if interediate_resnet_layer == 2:
            #in resnet18 mini_resnet_channels was 128
            backbone.layer2._modules['0']._modules['conv1'] = nn.Conv2d(original_dims, mini_resnet_channels, 
                                                                        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            backbone.layer2._modules['0']._modules['bn1'] = FrozenBatchNorm2d(mini_resnet_channels)
            backbone.layer2._modules['0']._modules['conv2'] = nn.Conv2d(mini_resnet_channels, out_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer2._modules['0']._modules['bn2'] = FrozenBatchNorm2d(out_dims)

            backbone.layer2._modules['0']._modules['downsample']._modules['0'] = nn.Conv2d(original_dims, out_dims, kernel_size=(1, 1), stride=(2, 2), bias=False)
            backbone.layer2._modules['0']._modules['downsample']._modules['1'] = FrozenBatchNorm2d(out_dims)

            backbone.layer2._modules['1']._modules['conv1'] = nn.Conv2d(out_dims, mini_resnet_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer2._modules['1']._modules['bn1'] = FrozenBatchNorm2d(mini_resnet_channels)
            backbone.layer2._modules['1']._modules['conv2'] = nn.Conv2d(mini_resnet_channels, out_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer2._modules['1']._modules['bn2'] = FrozenBatchNorm2d(out_dims)
        
        
        
        elif interediate_resnet_layer == 3:
            
            mini_resnet_channels = 128  #in resnet18 mini_resnet_channels was 256
            backbone.layer3._modules['0']._modules['conv1'] = nn.Conv2d(original_dims*2, mini_resnet_channels, 
                                                                        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            backbone.layer3._modules['0']._modules['bn1'] = FrozenBatchNorm2d(mini_resnet_channels)
            backbone.layer3._modules['0']._modules['conv2'] = nn.Conv2d(mini_resnet_channels, out_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer3._modules['0']._modules['bn2'] = FrozenBatchNorm2d(out_dims)

            backbone.layer3._modules['0']._modules['downsample']._modules['0'] = nn.Conv2d(original_dims*2, out_dims, kernel_size=(1, 1), stride=(2, 2), bias=False)
            backbone.layer3._modules['0']._modules['downsample']._modules['1'] = FrozenBatchNorm2d(out_dims)

            backbone.layer3._modules['1']._modules['conv1'] = nn.Conv2d(out_dims, mini_resnet_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer3._modules['1']._modules['bn1'] = FrozenBatchNorm2d(mini_resnet_channels)
            backbone.layer3._modules['1']._modules['conv2'] = nn.Conv2d(mini_resnet_channels, out_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer3._modules['1']._modules['bn2'] = FrozenBatchNorm2d(out_dims)
        
        
        
        elif interediate_resnet_layer == 4:
            mini_resnet_channels = 64 #in resnet18 mini_resnet_channels was 512
            backbone.layer4._modules['0']._modules['conv1'] = nn.Conv2d(original_dims*4, mini_resnet_channels, 
                                                                        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            backbone.layer4._modules['0']._modules['bn1'] = FrozenBatchNorm2d(mini_resnet_channels)
            backbone.layer4._modules['0']._modules['conv2'] = nn.Conv2d(mini_resnet_channels, out_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer4._modules['0']._modules['bn2'] = FrozenBatchNorm2d(out_dims)

            backbone.layer4._modules['0']._modules['downsample']._modules['0'] = nn.Conv2d(original_dims*4, out_dims, kernel_size=(1, 1), stride=(2, 2), bias=False)
            backbone.layer4._modules['0']._modules['downsample']._modules['1'] = FrozenBatchNorm2d(out_dims)

            backbone.layer4._modules['1']._modules['conv1'] = nn.Conv2d(out_dims, mini_resnet_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer4._modules['1']._modules['bn1'] = FrozenBatchNorm2d(mini_resnet_channels)
            backbone.layer4._modules['1']._modules['conv2'] = nn.Conv2d(mini_resnet_channels, out_dims, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone.layer4._modules['1']._modules['bn2'] = FrozenBatchNorm2d(out_dims)
        
        return backbone

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm. 
    I've changed the first conv layer parameters accordingly to 
    https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/5"""
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 num_input_channels:int,
                 interediate_resnet_layer:int, 
                 n_channels_backbone_out:int, 
                 last_layer_intermediate_channels:int):
        
        mini_resnet18 = False
        
        if name == 'resnet50':
            dilation = dilation
            num_channels = 2048
        elif name in ['resnet18', 'resnet34']:
            dilation = False #only supported for resnet50
            num_channels = 512
        elif name == 'mini_resnet18':
            name = 'resnet18'
            dilation = False #only supported for resnet50
            num_channels = 512
            mini_resnet18 = True
        else:
            raise ValueError('not supported')
         
        #build resnet from scratch
        #backbone = torchvision.models.ResNet(block = torchvision.models.resnet.BasicBlock, 
        #                      layers = [1,1,1,1], replace_stride_with_dilation=[False, False, False], #[2,2,2,2] is resnet18
        #                      norm_layer=FrozenBatchNorm2d)
        
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d) #pretrained=is_main_process()
        
        
        backbone.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    
        if mini_resnet18:
            backbone = reduce_resnet18(backbone, n_channels_backbone_out, interediate_resnet_layer = interediate_resnet_layer,  mini_resnet_channels = last_layer_intermediate_channels)
            num_channels = n_channels_backbone_out
        
        super().__init__(backbone, train_backbone, num_channels, interediate_resnet_layer, n_channels_backbone_out, last_layer_intermediate_channels)
        
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):        
        #self[0] is Backbone(), self[1] is PositionEmbeddingSine()
        #PositionEmbeddingSine() takes as input x, which is the output of Backbone.
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype)) #PositionEmbeddingSine() takes as input x, which is the output of Backbone.
        return out, pos
    
def build_backbone(args):
    num_input_channels = args.proj_module_N_channels*2
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    backbone = Backbone(args.backbone, train_backbone, args.dilation, 
                        num_input_channels, args.interediate_resnet_layer,
                        args.n_channels_backbone_out, args.last_layer_intermediate_channels)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
