from torch import nn, Tensor
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM
    
class NTProjectionModule(nn.Module):
    """ 
    This is the projection module for BERT
    """
    def __init__(self, proj_module_N_channels):
        """ 
        Parameters:
            proj_module_N_channels: the number of output channels.
        """
        super().__init__()
        nt_dim = EMBEDDING_DIM
        self.conv1d = nn.Conv1d(in_channels=nt_dim, out_channels=proj_module_N_channels, kernel_size=1)
        #self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(nt_dim)

    def forward(self, xs: Tensor):
        xs = self.bn(xs)
        xs = self.conv1d(xs)
        return xs



def build_projection_module_nt(args):
    nt_module = NTProjectionModule(args.proj_module_N_channels)
    return nt_module