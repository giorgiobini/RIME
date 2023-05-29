import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import itertools
import numpy as np
from . import box_ops

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

def create_contact_matrix_for_masks(rna0, rna1):
    """
    I will use the following notation:
    # b is the batch_size
    # N is the rna1 length
    # M is the rna2 length
    
    Args:
        rna0 (torch.Tensor): Batch containing the first masked rna of the pair with shape (b * N * 1)
        rna1 (torch.Tensor): Batch containing the second masked rna of the pair with shape (b * M * 1)
    Returns:
        contact tensor (torch.Tensor): Contact matrix with shape (b * N * M)
    """
    #shapes rna0, rna1 -> torch.Size([b, N, 1]) torch.Size([b, M, 1])
    rna0_int, rna1_int = prepare_rna_mask(rna0), prepare_rna_mask(rna1)
    #shapes rna0_int, rna0_int -> torch.Size([b, 1, N]) torch.Size([b, 1, M])
        
    # rna0.unsqueeze(3) is (b, 1 ,N, 1), rna1.unsqueeze(2) is (b, 1, 1, M)
    rna0_adj = rna0_int.unsqueeze(3) - torch.zeros(rna1_int.unsqueeze(2).shape, device = rna0.device)
    rna1_adj = rna1_int.unsqueeze(2) - torch.zeros(rna0_int.unsqueeze(3).shape,  device = rna0.device)
    # rna0_adj and rna1_adj are (b, 1, N, M)
    
    rna_contact = torch.cat([rna0_adj, rna1_adj],1) # rna_contact is (b, 2, N, M)
    rna_contact = rna_contact.sum(axis = 1) # rna_contact is (b, N, M)
    rna_contact = rna_contact > 0 #only False-False combinations are False. False-True I consider True, because it means one of the two RNA are padded in that position. Remember: False means no-padding
    
    return rna_contact

def prepare_rna_mask(rna_mask):
    """
    Some operations we do with tensors inside the create_contact_matrix() function are not allowed if the tensor are boolean. 
    So I need to convert tensor in boolean integers.
    
    As for the permutation operation, I do it because originally the masked tensors are of shape (batch_size, rna_length, 1) 
    and the last dimension was just fake (only needed for working with NestedTensors). 
    Now I need that dimension in the create_contact_matrix function in order to indicate the number of channels.
    Args:
        rna_mask: torch tensor of boolean.
    """
    rna_int = (rna_mask > 0).type(torch.int64)
    rna_int = rna_int = rna_int.transpose(1, 2)
    return rna_int

def hydrogen_bond_contact_matrix(cdna1, cdna2):
        """
        Args:
            cdna1 (str): the sequence of the first rna
            cdna2 (str): the sequence of the second rna
        Returns:
            contact matrix (np.array): Hydrogen bond contact matrix of shape (N * M)
        """
        hydrogen_bonds = {'AA':0,'AC':0,'AG':0,'AT':2,
                          'CA':0,'CC':0,'CG':3,'CT':0,
                          'GA':0,'GC':3,'GG':0,'GT':1,
                          'TA':2,'TC':0,'TG':1,'TT':0
                         }
        cdna1_list = [b for b in cdna1]
        cdna2_list = [b for b in cdna2]

        interaction_list = list(itertools.product(cdna1_list, cdna2_list))
        list_to_be_mapped = [''.join(b) for b in interaction_list]
        mapped_list = list(map(hydrogen_bonds.get, list_to_be_mapped))
        return np.array(mapped_list).reshape(len(cdna1_list), len(cdna2_list)) 

def plot_contact_matrix(cdna1, cdna2, list_of_boxes,  crop_bbox = [], list_of_predictions = [], list_of_probabilities = [], plot_hbonds = False, plot_geoms = True, cmap_box = plt.cm.twilight_shifted):
    """
    Args:
        cdna1 (str): the sequence of the first rna
        cdna2 (str): the sequence of the second rna
        list_of_boxes (list): list of the bounding boxes in this matrix
        crop_bbox (list): if provided, it plots a black rectangle where I am going to crop the rna_rna pair
        list_of_predictions (list): if provided, it plots the predicted the bounding boxes 
        list_of_probabilities (list): if provided, it represents the probablities associated to all the predictions. It plots the intensity of the probabilities.
        plot_hbonds (bool): if true, hydrogen_bonds will be plotted.
        plot_geoms (bool): whether or not to plot the following points:
            - (x,y) the green point;
            - (xc, yc) the red star;
            - (x+w, y+h) the yellow point
    Returns:
        plot: matplotilb plot of the hydrogen bond matrix with all the bboxes 
    """

    if plot_hbonds:
        fig, ax = plt.subplots(figsize=(3, 1))
        fig.subplots_adjust(bottom=0.5)
        cmap = mpl.colors.ListedColormap(['white', '#F1DBF7', '#BA93C5', '#6E5B73'])
        bounds = [0.0, 1.0, 2.0, 3.0, 4.0]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        ticks=[0, 1, 2, 3.0],
                                        orientation='horizontal')
        cb.set_label('Hydrogen bonds')
    else: 
        cmap = mpl.colors.ListedColormap(['white'])
        
    if len(list_of_probabilities)>0:
        assert len(list_of_predictions) == len(list_of_probabilities)
        fig2, ax2 = plt.subplots(figsize=(1, 3))
        cmap_box = cmap_box #plt.cm.Purples
        bounds = np.arange(101)
        norm = mpl.colors.BoundaryNorm(bounds, cmap_box.N)
        cb_b = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_box,
                                    norm=norm,
                                    orientation='vertical')
        cb_b.set_label('Probabilities')

    plt.rcParams["figure.figsize"] = (int(len([b for b in cdna1])/5), int(len([b for b in cdna2])/5))
    if plot_hbonds:
        plt.matshow(hydrogen_bond_contact_matrix(cdna1, cdna2).T, cmap = cmap);
    else:
        plt.matshow(np.zeros((len(cdna1), len(cdna2))).T, cmap = cmap);
    plt.xticks(range(len([b for b in cdna1])), [b for b in cdna1], size='small')
    plt.yticks(range(len([b for b in cdna2])), [b for b in cdna2], size='small')
    ax = plt.gca()
    colors = itertools.cycle('ygrcmk')
    
    for i, (x, y, w, h) in enumerate(list_of_boxes):
        ax.add_patch(plt.Rectangle((x, y), w-1, h-1, fill=False, color=next(colors), linewidth=4))
        if plot_geoms:
            plt.plot([x], [y], marker='o', markersize=10, color='green')
            plt.plot([x+w], [y+h], marker='o', markersize=10, color='yellow')
            xc, yc, w, h = box_ops.box_xyxy_to_cxcywh(torch.tensor([x, y, x+w, y+h]))
            plt.plot([xc], [yc], marker='*', markersize=10, color='red')
    
    if len(crop_bbox) == 4:
        X1, Y1, W_crop, H_crop = crop_bbox
        ax.add_patch(plt.Rectangle((X1, Y1), W_crop-1, H_crop-1, fill=False, color='black', linestyle = '--', linewidth = 7))
        
    if (len(list_of_predictions)>0)&(len(list_of_probabilities)==0):
        for i, (x, y, w, h) in enumerate(list_of_predictions):
            ax.add_patch(plt.Rectangle((x, y), w-1, h-1, fill=False, linestyle = '-.', color='navy', linewidth=4))
    
    elif (len(list_of_predictions)>0)&(len(list_of_probabilities)>0):
        for i, (x, y, w, h) in enumerate(list_of_predictions):
            intensity = list_of_probabilities[i]
            ax.add_patch(plt.Rectangle((x, y), w-1, h-1, fill=False, linestyle = '-.', color=cmap_box(intensity), linewidth=4))
            
    plt.show()
    
import sys
sys.path.append('../')
from dataset.data import plot_sample, Sample, BBOX

def new_plot_contact_matrix(cdna1, cdna2, list_of_boxes, crop_bbox, plot_cdna = True):
    interacting = True if len(list_of_boxes)>0 else False
    sample = Sample(gene1='gene1', gene2='gene2', 
               bbox = BBOX.from_xyhw(
                                    x=int(crop_bbox[0]),
                                    y=int(crop_bbox[1]),
                                    width=int(crop_bbox[2]),
                                    height=int(crop_bbox[3]),
               ),
               policy = '',
               interacting = interacting,
               seed_interaction_bbox = BBOX(0,0,0,0),
               all_couple_interactions =[
                                           {'x1': c[0], 'y1': c[1], 'w': c[2], 'h': c[3]} 
                                            for c in list_of_boxes
                                        ],
               gene1_info = {'cdna':cdna1},
               gene2_info = {'cdna':cdna2})
    sample_fig = plot_sample(sample, plot_cdna = plot_cdna)
    sample_fig.show(dpi=600)