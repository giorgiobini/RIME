import torch
import torch.nn.functional as F
from torch import nn
from util.misc import nested_tensor_from_tensor_list, NestedTensor

class BinaryClassificationModule(nn.Module): 
    def __init__(self, hidden_dim, dropout_rate):
        super(BinaryClassificationModule, self).__init__()
        self.bn1 = nn.BatchNorm2d(hidden_dim) # try batchnorm
        # Convolution Layers
        fc_dim1 = int(hidden_dim/2)
        fc_dim2 = 16
        self.conv1 = nn.Conv2d(hidden_dim, fc_dim1, kernel_size=2)
        self.drop1 = nn.Dropout(dropout_rate)
        # compute the flatten size
        self.fc1 = nn.Linear(fc_dim1, fc_dim2)
        self.fc2 = nn.Linear(fc_dim2, 2)
        self.drop2 = nn.Dropout(dropout_rate)
        self.gradients = None
        
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, X):
        X = self.bn1(X)
        X = self.drop1(X)
        # Convolution & Pool Layers
        X = self.conv1(X)
        
        if X.requires_grad:
            # register the hook
            h = X.register_hook(self.activations_hook)
        
        X = F.relu(X)
        
        X = F.max_pool2d(X, kernel_size=X.size()[2:]) #global max pooling
        #print(X.shape) -> torch.Size([1, 128, 1, 1])
        X = torch.squeeze(X, -1)
        X = torch.squeeze(X, -1) 
        # Its ugly, but if I do like this I am sure I squeeze only the last 2 dims, so that if I have batch_size == 1, it still works.
        #print(X.shape) -> torch.Size([1, 128])
        
        X = self.drop2(X) 
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        
        #print(X.shape) -> torch.Size([1, 2])
        return F.log_softmax(X, dim=1), X


class BinaryDETR(nn.Module):
    """ This is the module that performs binary classification """
    def __init__(self, backbone, transformer, dropout_rate):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.binary_class_module = BinaryClassificationModule(hidden_dim, dropout_rate)
        
    def get_transformer_output(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
        #print(samples.decompose()[0].shape) #-> torch.Size([1, 32, 507, 507])
        
        features, pos = self.backbone(samples)
        
        
        #features is a list of length 1
        #features[0] is a NestedTensor of length 2 
        #print(features[0].decompose()[0].shape) #-> torch.Size([1, args.n_channels_backbone_out, reduced_height, reduced_width])
        
        
        #pos is a list of length 1
        #pos[0] is a Tensor
        #print(pos[0].shape) #-> torch.Size([1, 64, reduced_height, reduced_width])
        
        src, mask = features[-1].decompose()
        assert mask is not None
        
        
        #memory = self.transformer(self.input_proj(src), mask, torch.empty((0, self.hidden_dim), dtype=torch.float64, device = src.device), pos[-1])[1]
        
        memory = self.transformer(self.input_proj(src), mask, None, pos[-1])
        
        #print(memory.shape) #-> torch.Size([batch_size, 64, reduced_height, reduced_width])

        return memory
    
    def forward(self, samples: NestedTensor):
        """
        The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        
        memory = self.get_transformer_output(samples)
        
        #print(memory.shape) #-> torch.Size([batch_size, 64, reduced_height, reduced_width])

        output = self.binary_class_module(memory)
        out = {'cnn_output':output[1], 'probas':output[0]}
        return out
    
def build_image_cl(backbone, transformer, dropout_rate):
    return BinaryDETR(backbone, transformer, dropout_rate)