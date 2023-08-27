import torch
from torch import nn, Tensor
import sys
import os
import torch.nn.functional as F
from util.contact_matrix import create_contact_matrix
from .mlp import build as build_top_classifier
from .bert import build_bert

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM_BERT

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
        dnabert_dim = EMBEDDING_DIM_BERT
        self.conv1d = nn.Conv1d(in_channels=dnabert_dim, out_channels=proj_module_N_channels, kernel_size=1)
        #self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(dnabert_dim)

    def forward(self, xs: Tensor):
        xs = self.bn(xs)
        xs = self.conv1d(xs)
        return xs

class SmallCNN(nn.Module):
    def __init__(self, k, n_channels1 = 16, n_channels2 = 32):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(k, n_channels1, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(n_channels1, n_channels2, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=1, stride=1)
        self.tanh = nn.Tanh()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_channels2, n_channels2)
        self.output_layer = nn.Linear(n_channels2, 2)
        self.gradients1 = None
        self.gradients2 = None

    # hook for the gradients of the activations
    def activations_hook1(self, grad):
        self.gradients1 = grad
        
     # hook for the gradients of the activations
    def activations_hook2(self, grad):
        self.gradients2 = grad

    def forward(self, x):
        #print(x.shape) --> torch.Size([batch, k, lenrna1, lenrna2])
        x = self.conv1(x)
        if x.requires_grad:
            # register the hook
            h = x.register_hook(self.activations_hook1)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv2(x)
        if x.requires_grad:
            # register the hook
            h = x.register_hook(self.activations_hook2)
        x = self.relu(x)
        x = self.maxpool2d(x)
        #print(x.shape) --> torch.Size([batch, n_channels2, reduced_lenrna1, reduced_lenrna2])
        x = self.global_avg_pool(x)
        #print(x.shape) --> torch.Size([batch, n_channels2, 1, 1])
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print(x.shape) --> torch.Size([batch, n_channels2])
        x = self.fc(x)
        x = self.tanh(x)
        x = self.output_layer(x)
        return x
    
class BinaryClassifierBERT(nn.Module):
    def __init__(self, bert, projection_module, small_cnn):
        super().__init__()
        self.bert = bert
        self.projection_module = projection_module
        self.small_cnn = small_cnn

    def obtain_bert_embeddings(self, rna1, rna2):
        rna1 = self.bert(rna1) 
        rna1 = rna1[0][:, 1:-1,:] #exclude the first CLS and the last SEP chars
        rna2 = self.bert(rna2) 
        rna2 = rna2[0][:, 1:-1,:] #exclude the first CLS and the last SEP chars
        #shape rna2 ->  torch.Size([b, len_rna1 - 5, 768])
        #shape rna2 ->  torch.Size([b, len_rna2 - 5, 768])
        rna1 = rna1.permute(0, 2, 1)
        rna2 = rna2.permute(0, 2, 1)
        #shape rna2 ->  torch.Size([b, 768, len_rna1 - 5])
        #shape rna2 ->  torch.Size([b, 768, len_rna2 - 5])
        return rna1, rna2

    def project_embeddings(self, rna1, rna2):
        #shape rna1 ->  torch.Size([b, len_rna1 - 5, 768])
        #shape rna2 ->  torch.Size([b, len_rna2 - 5, 768])
        rna1 = self.projection_module(rna1)
        rna2 = self.projection_module(rna2)
        #shapes rna1, rna2 -> torch.Size([b, d, len_rna1 - 5]) torch.Size([b, d, len_rna2 - 5])
        return rna1, rna2


    # method for the activation exctraction
    def get_activations1(self, rna1, rna2):
        rna1, rna2 = self.obtain_bert_embeddings(rna1, rna2)
        rna1, rna2 = self.project_embeddings(rna1, rna2)
        X = create_contact_matrix(rna1, rna2) 
        del rna1, rna2
        X = self.small_cnn.conv1(X)
        return X
    
    # method for the activation exctraction
    def get_activations2(self, rna1, rna2):
        rna1, rna2 = self.obtain_bert_embeddings(rna1, rna2)
        rna1, rna2 = self.project_embeddings(rna1, rna2)
        X = create_contact_matrix(rna1, rna2) 
        del rna1, rna2
        X = self.small_cnn.conv1(X)
        X = self.small_cnn.relu(X)
        X = self.small_cnn.maxpool2d(X)
        X = self.small_cnn.conv2(X)
        return X
    
    # method for the gradient extraction
    def get_activations_gradient1(self):
        return self.small_cnn.gradients1
    
    # method for the gradient extraction
    def get_activations_gradient2(self):
        return self.small_cnn.gradients2
    
    def forward(self, rna1, rna2):
        rna1, rna2 = self.obtain_bert_embeddings(rna1, rna2)
        rna1, rna2 = self.project_embeddings(rna1, rna2)
        #shapes rna1, rna2 -> torch.Size([b, d, len_rna1 - 5]) torch.Size([b, d, len_rna2 - 5])
        contact_matrix = create_contact_matrix(rna1, rna2) 
        #shape create_contact_matrix(rna1, rna2) -> torch.Size([b, 2d,  len_rna1 - 5,  len_rna2 - 5])
        del rna1, rna2
        binary_output = self.small_cnn(contact_matrix)
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
    
    
def build_projection_module(args):
    bert_module = BERTProjectionModule(args.proj_module_N_channels)
    return bert_module
    
def build(args, bert_pretrained_path):
    
    bert = build_bert(args, bert_pretrained_path)

    projection_module = build_projection_module(args)
    
    small_cnn = SmallCNN(
        k = args.proj_module_N_channels*2, n_channels1 = args.n_channels1_cnn, n_channels2 = args.n_channels2_cnn
    )
    
    model = BinaryClassifierBERT(bert, projection_module, small_cnn)
    
    return model