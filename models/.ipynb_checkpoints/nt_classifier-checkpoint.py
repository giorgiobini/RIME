import torch
from torch import nn
import torch.nn.functional as F
from util.contact_matrix import create_contact_matrix, create_contact_matrix_for_masks
from util.misc import NestedTensor
from .projection_module import build_projection_module_nt
from .top_classifier import build_top_classifier
    
class BinaryClassifierNT(nn.Module):
    def __init__(self, nt_projection_module, top_classifier, mini_batch_size):
        super().__init__()
        self.nt_projection_module = nt_projection_module
        self.top_classifier = top_classifier
        self.mini_batch_size = mini_batch_size
        
    def obtain_mlp_output(self, rna1, rna2):
        # shapes rna1, rna2 -> torch.Size([batch_size, 2560, len_rna1]) torch.Size([batch_size, 2560, len_rna2])
        
        rna1 = self.nt_projection_module(rna1)
        rna2 = self.nt_projection_module(rna2)
        
        # shapes rna1, rna2 -> torch.Size([batch_size, d, len_rna1]) torch.Size([batch_size, d, len_rna2])

        batch_size, d, len_rna1 = rna1.size()
        _, _, len_rna2 = rna2.size()
        

        mlp_outputs = []
        for i in range(0, batch_size):
            mlp_outputs = []
            rna1_batch = rna1[i, :, :].unsqueeze(0) # shape rna1_batch -> torch.Size([1, d, len_rna1])
            rna2_batch = rna2[i, :, :].unsqueeze(0) # shape rna2_batch -> torch.Size([1, d, len_rna2])
            
            contact_matrix = create_contact_matrix(rna1_batch, rna2_batch)
            # shape contact_matrix ->  torch.Size([1, 2d,  len_rna1,  len_rna2])
            del rna1_batch
            del rna2_batch

            contact_matrix = contact_matrix.view(1, 2*d, len_rna1 * len_rna2).squeeze(0).permute(1, 0)
            # shape contact_matrix -> torch.Size([len_rna1*len_rna2, 2d])
            
            output_top_classifier = self.top_classifier.forward(contact_matrix)
            # shape output_top_classifier -> torch.Size([len_rna1*len_rna2, k])
            del contact_matrix

            _, k = output_top_classifier.size()
            output_top_classifier = output_top_classifier.view(len_rna1, len_rna2, k)

            mlp_outputs.append(output_top_classifier.unsqueeze(0))

        output = torch.cat(mlp_outputs, dim=0)
        #shape output -> torch.Size([batch_size, len_rna1, len_rna2, k])
        
        return output
        
    def forward(self, rna1, rna2):
        output_contact_matrix = self.obtain_mlp_output(rna1, rna2)
        binary_output = #qui devo processare output_contact_matrix
        return binary_output

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

    losses = {'accuracy': accuracy, 'precision': precision, 'recall':recall, 'F2': F2, 'specificity': TNR, 'NPV': NPV} 
    return losses
    
    
    
def build(args):
    
    nt_projection_module = build_projection_module_nt(args)
    
    top_classifier = build_top_classifier(
        dropout_rate = args.dropout_binary_cl
    )
    
    model = BinaryClassifierNT(nt_projection_module, top_classifier)
    
    return model