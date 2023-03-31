from torch import nn, Tensor
import torch
from util.misc import NestedTensor

class BERTProjectionModule(nn.Module):
    """ 
    This is the projection module for BERT
    """
    def __init__(self, proj_module_N_channels):
        """ 
        Parameters:
            proj_module_N_channels: the number of output channels.
        """
        super().__init__()
        dnabert_dim = 768
        self.conv1d = nn.Conv1d(in_channels=dnabert_dim, out_channels=proj_module_N_channels, kernel_size=1)
        #self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(dnabert_dim)

    def forward(self, xs: Tensor):
        xs = self.bn(xs)
        xs = self.conv1d(xs)
        return xs
    
class SecondaryStructureProjectionModule(nn.Module):
    """ 
    This is the projection module for secondary structure 
    """
    def __init__(self, proj_module_N_channels, drop_secondary_structure):
        """ 
        Parameters:
            proj_module_N_channels: the number of output channels.
        """
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=3, out_channels=proj_module_N_channels, kernel_size=6) # with kernel_size=6 I will reach the desired dimension (we work with 6mers)
        #self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(3)
        self.drop_secondary_structure = drop_secondary_structure

    def forward(self, xs: Tensor):
        if self.drop_secondary_structure:
            xs = xs * 0
        xs = self.bn(xs)
        xs = self.conv1d(xs)
        return xs


def build_projection_module(args):
    bert_module = BERTProjectionModule(args.proj_module_N_channels)
    secondary_structure_module = SecondaryStructureProjectionModule(args.proj_module_secondary_structure_N_channels, args.drop_secondary_structure)
    return bert_module, secondary_structure_module