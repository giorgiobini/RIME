from torch import nn
from util.contact_matrix import create_contact_matrix, create_contact_matrix_for_masks
from .detr import build_detr
from .projection_module import build_projection_module
from util.misc import NestedTensor

class DETRNARNA(nn.Module):
    """ This connects the DETR module with the projection_module"""
    def __init__(self, projection_module, detr):
        """

        """
        super().__init__()
        self.projection_module = projection_module
        self.detr = detr

    def forward(self, rna1, rna2):
        """ 
        """
        #shapes rna1.tensors, rna2.tensors -> torch.Size([b, 768, N, 1]) torch.Size([b, 768, M, 1])
        rna1 = self.projection_module(rna1)
        rna2 = self.projection_module(rna2)
        #shapes rna1, rna2 -> torch.Size([b, 16, N, 1]) torch.Size([b, 16, M, 1])
        contact_matrix_tensors = create_contact_matrix(rna1.tensors.squeeze(-1), rna2.tensors.squeeze(-1)) #now I can delete the uneuseful dimension from both rnas (I was using this extra dim only to deal with nested tensors)
        #shape contact_matrix_tensors -> torch.Size([b, 2d, N, M])
        
        contact_matrix_masks = create_contact_matrix_for_masks(rna1.mask, rna2.mask)
        #shape contact_matrix_masks -> torch.Size([b, N, M])
        
        contact_matrix = NestedTensor(contact_matrix_tensors, contact_matrix_masks)
        out = self.detr.forward(contact_matrix)
        return out
    
def build(args):
    
    projection_module = build_projection_module(args)
    
    detr, criterion, postprocessors = build_detr(args)

    model = DETRNARNA(projection_module, detr)
    
    return model, criterion, postprocessors