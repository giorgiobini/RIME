import torch
from torch import nn
import torch.nn.functional as F
from util.contact_matrix import create_contact_matrix, create_contact_matrix_for_masks
from util.misc import NestedTensor
from .backbone import build_backbone
from .transformer import build_transformer
from .projection_module import build_projection_module, build_projection_module_nt
from .image_classifier import build_image_cl

class BinaryClassifierRNARNA(nn.Module):
    """ This connects the DETR module with the projection_module"""
    def __init__(self, bert, bert_projection_module, ss_projection_module, image_cl):
        """

        """
        super().__init__()
        self.bert = bert
        self.bert_projection_module = bert_projection_module
        self.ss_projection_module = ss_projection_module
        self.image_cl = image_cl
        
    def create_contact_matrix(self, rna1, rna2):
        rna1ss, rna1_bert = rna1
        rna2ss, rna2_bert = rna2
        #shapes rna1ss, rna2ss -> torch.Size([b, 3, len_rna1]),  torch.Size([b, 3, len_rna2])
        
        del rna1
        del rna2
        
        len_rna1_ss = rna1ss.shape[-1]
        len_rna2_ss = rna2ss.shape[-1]

        len_rna1 = rna1_bert.tensors.shape[1] + 3
        len_rna2 = rna2_bert.tensors.shape[1] + 3
        
        
        if (len_rna1 > 512)|(len_rna2 > 512):
              raise Exception(f"RNA length can be 512 nucleotides or less")
        
        
        contact_matrix_masks = create_contact_matrix_for_masks(rna1_bert.mask, rna2_bert.mask)
        #shape contact_matrix_masks -> torch.Size([b, len_rna1-3, len_rna2-3])

        
        rna1_bert = self.bert(rna1_bert.tensors) 
        rna1_bert = rna1_bert[0][:, 1:-1,:] #exclude the first CLS and the last SEP chars
        rna2_bert = self.bert(rna2_bert.tensors) 
        rna2_bert = rna2_bert[0][:, 1:-1,:] #exclude the first CLS and the last SEP chars
        #shape rna1_bert ->  torch.Size([b, len_rna1 - 5, 768])
        #shape rna2_bert ->  torch.Size([b, len_rna2 - 5, 768])
        
        rna1_bert = self.bert_projection_module(rna1_bert.permute(0,2,1))
        rna2_bert = self.bert_projection_module(rna2_bert.permute(0,2,1))
        #shapes rna1, rna2 -> torch.Size([b, d, len_rna1 - 5]) torch.Size([b, d, len_rna2 - 5])
        
        
        rna1ss = self.ss_projection_module(rna1ss)
        rna2ss = self.ss_projection_module(rna2ss)
        #shapes rna1ss, rna2ss -> torch.Size([b, k, len_rna1 - 5]) torch.Size([b, k, len_rna2 - 5])

        contact_matrix = NestedTensor(torch.cat(
                (create_contact_matrix(rna1_bert, rna2_bert), #shape create_contact_matrix(rna1_bert, rna2_bert) -> torch.Size([b, 2d,  len_rna1 - 5,  len_rna2 - 5])
                create_contact_matrix(rna1ss, rna2ss)) ,#shape create_contact_matrix(rna1ss, rna2ss) -> torch.Size([b, 2k, len_rna1 - 5,  len_rna2 - 5])
                1), contact_matrix_masks)
        return contact_matrix
        
    def forward(self, rna1, rna2):
        """ 
        """
        
        contact_matrix = self.create_contact_matrix(rna1, rna2)
        
        out = self.image_cl.forward(contact_matrix)
        return out
    
    # method for the activation exctraction
    def get_activations(self, rna1, rna2):
        X = self.create_contact_matrix(rna1, rna2)
        X = self.image_cl.get_transformer_output(X)
        X = self.image_cl.binary_class_module.bn1(X)
        X = self.image_cl.binary_class_module.conv1(X)
        return X
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.image_cl.binary_class_module.gradients
    
class BinaryClassifierNT(nn.Module):
    """ This connects the NT module with the projection_module"""
    def __init__(self, nt_projection_module, image_cl):
        """

        """
        super().__init__()
        self.nt_projection_module = nt_projection_module
        self.image_cl = image_cl
        
    def create_contact_matrix(self, rna1, rna2):
        
        contact_matrix_masks = create_contact_matrix_for_masks(rna1.mask, rna2.mask) 
        
        #shapes rna1.tensors, rna2.tensors -> torch.Size([b, 2560, len_rna1]) torch.Size([b, 2560, len_rna2])
        
        rna1 = self.nt_projection_module(rna1.tensors)
        rna2 = self.nt_projection_module(rna2.tensors)
        #shapes rna1, rna2 -> torch.Size([b, d, len_rna1]) torch.Size([b, d, len_rna2])

        contact_matrix = create_contact_matrix(rna1, rna2), #shape create_contact_matrix(rna1, rna2) -> torch.Size([b, 2d,  len_rna1,  len_rna2])
        
        contact_matrix = NestedTensor(
            create_contact_matrix(rna1, rna2), 
            contact_matrix_masks
        )
        
        return contact_matrix
    
    
#      def create_contact_matrix(self, rna1, rna2):
#         # WITHOUT USING NESTED TESOR
        
#         #shapes rna1, rna2 -> torch.Size([b, 2560, len_rna1]) torch.Size([b, 2560, len_rna2])
        
#         rna1 = self.nt_projection_module(rna1)
#         rna2 = self.nt_projection_module(rna2)
#         #shapes rna1, rna2 -> torch.Size([b, d, len_rna1]) torch.Size([b, d, len_rna2])

#         contact_matrix = create_contact_matrix(rna1, rna2), #shape create_contact_matrix(rna1, rna2) -> torch.Size([b, 2d,  len_rna1,  len_rna2])
        
#         return contact_matrix
        
    def forward(self, rna1, rna2):
        """ 
        """
        
        contact_matrix = self.create_contact_matrix(rna1, rna2)
        
        out = self.image_cl.forward(contact_matrix)
        return out
    
    # method for the activation exctraction
    def get_activations(self, rna1, rna2):
        X = self.create_contact_matrix(rna1, rna2)
        X = self.image_cl.get_transformer_output(X)
        X = self.image_cl.binary_class_module.bn1(X)
        X = self.image_cl.binary_class_module.conv1(X)
        return X
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.image_cl.binary_class_module.gradients   
    
    
def obtain_predictions_ground_truth(outputs, targets):
    cnn_output = outputs['cnn_output']

    ground_truth = torch.cat([ torch.tensor([ t['interacting'] ]) 
                              for t in targets])
    return cnn_output, ground_truth.to(cnn_output.device)

class SetCriterion(nn.Module):
    def __init__(self, losses, weight_dict):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        
    def loss_binary_ce(self, outputs, targets):
        """
        """
        cnn_output, ground_truth = obtain_predictions_ground_truth(outputs, targets)
        binary_ce = F.cross_entropy(cnn_output, ground_truth)
        
        losses = {'loss_ce': binary_ce}
        
        return losses
    
    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'binary_ce': self.loss_binary_ce,
            'metrics': self.metrics
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)
        
    def forward(self, outputs, targets):
        """ 
        """

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        return losses
    
    @torch.no_grad()
    def metrics(self, outputs, targets):
        """ 
        Compute accuracy, precision, recall, f1
        """
        cnn_output, ground_truth = obtain_predictions_ground_truth(outputs, targets) 
        
        losses =  calc_metrics(cnn_output, ground_truth)
        return losses
    
    
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
    device = torch.device(args.device)
    
    nt_projection_module = build_projection_module_nt(args)
    
    backbone = build_backbone(args)
    
    transformer = build_transformer(args)
    
    image_cl = build_image_cl(
        backbone,
        transformer, 
        dropout_rate = args.dropout_binary_cl
    )
    
    model = BinaryClassifierNT(nt_projection_module, image_cl)
    
    weight_dict = {'loss_ce': 1.0}
    
    losses = ['binary_ce', 'metrics']
    
    criterion =  SetCriterion(losses, weight_dict = weight_dict)
    
    criterion.to(device)
    postprocessors = {}

    
    return model, criterion, postprocessors