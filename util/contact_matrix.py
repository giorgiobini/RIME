import torch

def create_contact_matrix(rna0, rna1, rna_length_first = False):
    """
    I will use the following notation:
    # b is the batch_size
    # d is the embedding_dim
    # N is the rna1 length
    # M is the rna2 length
    
    Args:
        rna0 (torch.Tensor): Batch containing the first rna of the pair with shape (b * d * N)
        rna1 (torch.Tensor): Batch containing the second rna of the pair with shape (b * d * M)
        rna_length_first (bool): if true, then the tensors are with shape (b * N * d), (b * M * d)
    Returns:
        contact tensor (torch.Tensor): Contact matrix with shape (b * 2d * N * M)
    """
    if rna_length_first:
        # rna0 is (b,N,d), rna1 is (b,M,d)
        rna0 = rna0.transpose(1, 2)
        rna1 = rna1.transpose(1, 2)
        # rna0 is (b,d,N), rna1 is (b,d,M)
    
        
    # rna0.unsqueeze(3) is (b,d,N, 1), rna1.unsqueeze(2) is (b,d,1,M)
    rna0_adj = rna0.unsqueeze(3) - torch.zeros(rna1.unsqueeze(2).shape, device = rna0.device)
    rna1_adj = rna1.unsqueeze(2) - torch.zeros(rna0.unsqueeze(3).shape, device = rna1.device)
    # rna0_adj and rna1_adj are (b,d,N, M)
    
    rna_contact = torch.cat([rna0_adj, rna1_adj],1)
    # rna_contact is (b,2d,N, M)
    
    return rna_contact
