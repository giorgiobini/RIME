import torch
from torch import nn
import torch.nn.functional as F
from util.contact_matrix import create_contact_matrix, create_contact_matrix_for_masks
from util.misc import NestedTensor
from .projection_module import build_projection_module_nt
from .mlp import build as build_top_classifier

class SmallCNN(nn.Module):
    def __init__(self, k, n_channels1 = 16, n_channels2 = 32):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(k, n_channels1, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(n_channels1, n_channels2, kernel_size=1, stride=1)
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
    
class BinaryClassifierNT(nn.Module):
    def __init__(self, nt_projection_module, top_classifier, small_cnn, use_projection_module):
        super().__init__()
        self.nt_projection_module = nt_projection_module
        self.top_classifier = top_classifier
        self.small_cnn = small_cnn
        self.use_projection_module = use_projection_module
        
    def obtain_mlp_output(self, rna1, rna2):
        # shapes rna1, rna2 -> torch.Size([batch_size, 2560, len_rna1]) torch.Size([batch_size, 2560, len_rna2])
        
        
        if self.use_projection_module:
            rna1 = self.nt_projection_module(rna1)
            rna2 = self.nt_projection_module(rna2)
            # shapes rna1, rna2 -> torch.Size([batch_size, d, len_rna1]) torch.Size([batch_size, d, len_rna2])

        batch_size, d, len_rna1 = rna1.size()
        _, _, len_rna2 = rna2.size()
        

        # I do this for loop such that the contact matrix is not huge
        mlp_outputs = []
        for i in range(0, batch_size):
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
            #shape output_top_classifier -> torch.Size([len_rna1, len_rna2, k])

            mlp_outputs.append(output_top_classifier.unsqueeze(0))

        output = torch.cat(mlp_outputs, dim=0)
        #shape output -> torch.Size([batch_size, len_rna1, len_rna2, k])
        output = output.permute(0, 3, 1, 2)
        #shape output -> torch.Size([batch_size, k, len_rna1, len_rna2])
        return output


    def obtain_mlp_output_reduced_memory(self, rna1, rna2):
        # shapes rna1, rna2 -> torch.Size([batch_size, 2560, len_rna1]) torch.Size([batch_size, 2560, len_rna2])

        if self.use_projection_module:
            rna1 = self.nt_projection_module(rna1)
            rna2 = self.nt_projection_module(rna2)
            # shapes rna1, rna2 -> torch.Size([batch_size, d, len_rna1]) torch.Size([batch_size, d, len_rna2])

        batch_size, d, len_rna1 = rna1.size()
        _, _, len_rna2 = rna2.size()

        # I do this for loop such that the contact matrix is not huge
        mlp_outputs = []
        for i in range(0, batch_size):
            rna2_batch = rna2[i, :, :].unsqueeze(0)  # shape rna2_batch -> torch.Size([1, d, len_rna2])
            
            rna1_outputs = []
            for j in range(0, len_rna1):
                rna1_batch_j = rna1[i, :, j].unsqueeze(1).unsqueeze(0)  # shape rna1_batch -> torch.Size([1, d, 1])

                contact_matrix = create_contact_matrix(rna1_batch_j, rna2_batch)
                # shape contact_matrix -> torch.Size([1, 2d, 1, len_rna2])

                output_top_classifier = self.top_classifier.forward(contact_matrix.squeeze(2).squeeze(0).permute(1, 0))
                del contact_matrix

                # shape output_top_classifier -> torch.Size([len_rna2, k])
                rna1_outputs.append(output_top_classifier.unsqueeze(0))
                del output_top_classifier
            
            rna1_outputs = torch.cat(rna1_outputs, dim=0).unsqueeze(0)
            # shape rna1_outputs -> torch.Size([1, len_rna1, len_rna2, k])
            mlp_outputs.append(rna1_outputs)
            del rna1_outputs


        output = torch.cat(mlp_outputs, dim=0)
        # shape output -> torch.Size([batch_size, len_rna1, len_rna2, k])
        output = output.permute(0, 3, 1, 2)
        # shape output -> torch.Size([batch_size, k, len_rna1, len_rna2])

        return output

    # method for the activation exctraction
    def get_activations1(self, rna1, rna2):
        X = self.obtain_mlp_output_reduced_memory(rna1, rna2) #obtain_mlp_output, obtain_mlp_output_reduced_memory
        X = self.small_cnn.conv1(X)
        return X
    
    # method for the activation exctraction
    def get_activations2(self, rna1, rna2):
        X = self.obtain_mlp_output_reduced_memory(rna1, rna2) #obtain_mlp_output, obtain_mlp_output_reduced_memory
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
        output_contact_matrix = self.obtain_mlp_output_reduced_memory(rna1, rna2) #obtain_mlp_output, obtain_mlp_output_reduced_memory
        binary_output = self.small_cnn(output_contact_matrix)
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
    
    small_cnn = SmallCNN(
        k = args.output_channels_mlp, n_channels1 = args.n_channels1_cnn, n_channels2 = args.n_channels2_cnn
    )
    
    top_classifier = build_top_classifier(args)
    
    model = BinaryClassifierNT(nt_projection_module, top_classifier, small_cnn, args.use_projection_module)
    
    return model