import numpy as np
import torch
from lime.lime_base import LimeBase
import random
import pandas as pd
from . import contact_matrix
from . import misc as utils
import sys
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
from torch.autograd import Variable

sys.path.append("..")
from UFold_dependencies.running_ufold import get_seq_embeddings_seq_ori, from_matrix_to_dot

"""
-------------- L I M E -------------- L I M E -------------- L I M E --------------
"""

def get_cdna_slices(s):
    sample_bbox = s.bbox 
    cdna1, cdna2 = s.gene1_info['cdna'], s.gene2_info['cdna']
    x1, x2, y1, y2 = sample_bbox.x1, sample_bbox.x2, sample_bbox.y1, sample_bbox.y2
    return cdna1[x1:x2], cdna2[y1:y2]

def obtain_dot_string(seq, gene_id, model):
    seq_embeddings, seq_lens, seq_ori, seq_name = get_seq_embeddings_seq_ori(seq, gene_id)
    model_features = model.contact_net_prediction(seq_embeddings, seq_lens, seq_ori, seq_name)
    dot_string = from_matrix_to_dot(model_features)
    return dot_string

def rna_item_assignment(dot_br, dot_br_slice, x1, x2):
    #can be dot_br or cdna
    new_dot_br = dot_br[:x1] + dot_br_slice + dot_br[x2:]
    assert len(new_dot_br) == len(dot_br)
    return new_dot_br

def prepare_samples(cdna1_slice, cdna2_slice, s, UFoldFeatureExtractor):
    sample_bbox = s.bbox
    x1, x2, y1, y2 = sample_bbox.x1, sample_bbox.x2, sample_bbox.y1, sample_bbox.y2
    
    ss1, ss2 = s.gene1_info['dot_br'], s.gene2_info['dot_br']

    ss1_slice = obtain_dot_string(cdna1_slice, '', UFoldFeatureExtractor)
    ss2_slice = obtain_dot_string(cdna2_slice, '', UFoldFeatureExtractor)

    s.gene1_info['dot_br'] = rna_item_assignment(s.gene1_info['dot_br'], ss1_slice, x1, x2)
    s.gene2_info['dot_br'] = rna_item_assignment(s.gene2_info['dot_br'], ss2_slice, y1, y2)
    
    s.gene1_info['cdna'] = rna_item_assignment(s.gene1_info['cdna'], cdna1_slice, x1, x2)
    s.gene2_info['cdna'] = rna_item_assignment(s.gene2_info['cdna'], cdna2_slice, y1, y2)
    
    samples, targets = utils.collate_fn([s])
    
    return samples, targets

def model_prob_from_input(samples, model, device):
    rna1, rna2 = samples
    rna1[0] = rna1[0].to(device)
    rna2[0] = rna2[0].to(device)
    rna1[1].tensors = rna1[1].tensors.to(device)
    rna2[1].tensors = rna2[1].tensors.to(device)
    rna1[1].mask = rna1[1].mask.to(device)
    rna2[1].mask = rna2[1].mask.to(device)
    outputs = model(rna1, rna2)
    probability = outputs['cnn_output'].softmax(-1)[:, 1]
    return probability

def forward_func(cdna1_slice, cdna2_slice, s, model, UFoldFeatureExtractor, device):
    samples, targets = prepare_samples(cdna1_slice, cdna2_slice, s, UFoldFeatureExtractor)
    probability = model_prob_from_input(samples, model, device)[0]
    return probability

"""
"""

def calc_perc(cdna):
    tot = 0
    res = Counter(cdna)
    for n in res.keys():
        tot += res[n]
    for n in res.keys():
        res[n] = res[n]/tot
    return res #res is something like {'G': 0.3,, 'C': 0.3, 'T': 0.2, 'A': 0.2}
    
def single_perturbation(cdna, perc = {'G': 0.25, 'C': 0.25, 'T': 0.25, 'A': 0.25}):
    position = random.choices(range(len(cdna)))[0]
    old_nucleotide = cdna[position]
    nucleotide = old_nucleotide
    while nucleotide == old_nucleotide:
        nucleotide = random.choices(list(perc.keys()), weights=perc.values(), k = 1)[0]
        cdna = cdna[:position] + nucleotide + cdna[position+1:]
    return cdna

def single_random_perturbation(cdna):
    position = random.choices(range(len(cdna)))[0]
    nucleotide = random.choices(['G', 'C', 'T', 'A'], k = 1)[0]
    cdna = cdna[:position] + nucleotide + cdna[position+1:]
    return cdna

def single_perturbation_2_rna(cdna1_slice, cdna2_slice, perturb_with_prob = True):
    if perturb_with_prob:
        perc1 = calc_perc(cdna1_slice)
        perc2 = calc_perc(cdna2_slice)
        cdna1_slice_pert = single_perturbation(cdna1_slice, perc = perc1)
        cdna2_slice_pert = single_perturbation(cdna2_slice, perc = perc2)
    else:
        cdna1_slice_pert = single_random_perturbation(cdna1_slice)
        cdna2_slice_pert = single_random_perturbation(cdna2_slice)
    return cdna1_slice_pert, cdna2_slice_pert


def build_one_hot_contact_matrix(cdna1, cdna2):
    """
    Args:
        cdna1 (str): the sequence of the first rna
        cdna2 (str): the sequence of the second rna
    Returns:
        contact matrix (np.array): one hot contact matrix of shape (8 * N * M)
    """
    encoding = {'A':0,'C':1,'G':2,'T':3}

    cdna1_list = [b for b in cdna1]
    cdna2_list = [b for b in cdna2]

    mapped_list1 = list(map(encoding.get, cdna1_list))
    mapped_list2 = list(map(encoding.get, cdna2_list))
    rna1 = torch.nn.functional.one_hot(torch.tensor(np.array(mapped_list1)), num_classes=4)
    rna2 = torch.nn.functional.one_hot(torch.tensor(np.array(mapped_list2)), num_classes=4)
    rna1, rna2 = rna1.unsqueeze(0), rna2.unsqueeze(0) # add batch dimension
    cm = contact_matrix.create_contact_matrix(rna1, rna2, rna_length_first = True).squeeze()#.transpose(1, 2)
    return cm
    
def distance_between_matrices(cm1, cm2, _, **kwargs):
    """
    cm1: torch.tensor of size 8, len(rna1), len(rna2)
    cm2: torch.tensor of size 8, len(rna1), len(rna2)

    cosine map is a 2d binary matrix of shape  8, len(rna1), len(rna2).
        In every cell there is 1 if the 8-length-one-hot 
        vector in that position is the same for cm1 and cm2, 
        0 if is different. 

    0 < similarity < 1: 1 if cm1 == cm2 in every position, 0 if there is no match in every position. 
    
    AC - AG is more similar than AC - GG, because in the first case they share the A in the first position.
    AC - CA is 0 similarity
    
    dist = 1-similarity
    """

    cm1, cm2 = cm1.float(), cm2.float()

    cosine_map = F.cosine_similarity(cm1, cm2, dim = 0)

    similarity = cosine_map.sum()/cosine_map.flatten().shape[0]

    dist = 1-similarity

    return dist #torch.exp(-1 * (dist ** 2) / 2)

def distance_from_original_input(original_input: tuple, perturbed: tuple):
    
    cdna1_slice, cdna2_slice = original_input
    
    cdna1_slice_pert, cdna2_slice_pert = perturbed
    
    cm = build_one_hot_contact_matrix(cdna1_slice, cdna2_slice)

    cm_pert = build_one_hot_contact_matrix(cdna1_slice_pert, cdna2_slice_pert)

    return distance_between_matrices(cm, cm_pert, '')

def create1dvector(cdna1_slice, cdna2_slice, cdna1_slice_pert, cdna2_slice_pert):    
    cdna1d = elementwise_equality(cdna1_slice, cdna1_slice_pert)
    cdna2d = elementwise_equality(cdna2_slice, cdna2_slice_pert)
    
    
    cdna1d = torch.tensor(cdna1d).unsqueeze(-1).unsqueeze(0) #torch.Size([1, N, 1])
    cdna2d = torch.tensor(cdna2d).unsqueeze(-1).unsqueeze(0) #torch.Size([1, M, 1])

    cm = contact_matrix.create_contact_matrix(cdna1d, cdna2d, rna_length_first = True).squeeze().mean(dim = 0)
    #cm: torch.Size([N, M]), 
    #In every pixel you have the mean of the 2 channel (each channel is binary, 1 if the nucleotide is the same of the original sequence 0 else).
    #So, every pixel can be 0, 0.5 or 1. NB: if the original pixel is AG, and the alternate is AC the pixel of cm is 0.5. If the alternate is CG the pixel of cm is 0.5. So the surrogate model is not aware of the direction.
    #NB: if the original pixel is AG, and the alternate is GA the pixel of cm is 0.
    return cm.flatten().numpy()

def elementwise_equality(cdna_old, cdna_new):
    return (np.array(list(cdna_old)) == np.array(list(cdna_new))).astype(int)

def lime(cdna1_slice, cdna2_slice, n_iters, max_perturbations_per_iter, probability,s, model, UFoldFeatureExtractor, device):
    len1 = len(cdna1_slice)
    len2 = len(cdna2_slice)
    neighborhood_data = [np.expand_dims(np.ones(len1 * len2), 0)]
    neighborhood_labels = [float(probability)]
    distances = [0]
    for i in range(n_iters):
    
        n_pert = random.sample(range(max_perturbations_per_iter), k = 1)[0]    

        cdna1_slice_pert, cdna2_slice_pert = cdna1_slice, cdna2_slice

        for j in range(n_pert):
            cdna1_slice_pert, cdna2_slice_pert = single_perturbation_2_rna(cdna1_slice_pert, cdna2_slice_pert, perturb_with_prob = False)

        vector1d = create1dvector(cdna1_slice, cdna2_slice, cdna1_slice_pert, cdna2_slice_pert)
        neighborhood_data.append(np.expand_dims(vector1d, 0))

        probability_pert = forward_func(cdna1_slice_pert, cdna2_slice_pert, s, model, UFoldFeatureExtractor, device)
        neighborhood_labels.append(float(probability_pert))

        dist = distance_from_original_input((cdna1_slice, cdna2_slice), (cdna1_slice_pert, cdna2_slice_pert))
        distances.append(float(dist))
        
    base = LimeBase(kernel_fn = lambda x: [1-i for i in x], #from distance to similarity
                    verbose = True,
                    random_state=123)
    
    neighborhood_data = np.concatenate(neighborhood_data, axis = 0)

    label1 = np.expand_dims(np.array(neighborhood_labels), 0)
    label0 = 1 - label1
    neighborhood_labels = np.concatenate([label0, label1], axis = 0).T

    distances = np.array(distances)
    
    label = 1
    num_features = len1*len2

    expl = base.explain_instance_with_data(neighborhood_data, 
                                           neighborhood_labels, 
                                           distances, 
                                           1, 
                                           num_features)
    
    res = expl[1]
    score = expl[2]
    assert len(res) == num_features
    coeff = np.array(pd.DataFrame.from_dict(res).sort_values(0).reset_index(drop = True)[1])
    
    expl_matrix = np.abs(coeff.reshape(len1, len2))
    
    return expl_matrix, score


def plot_matrix(matrix, cdna1, cdna2, list_of_boxes,  crop_bbox = [], list_of_predictions = [], cmap = 'viridis'):
    """
    Args:
        matrix (np.array): lime continuous matrix of shape N * M
        cdna1 (str): the sequence of the first rna of len N
        cdna2 (str): the sequence of the second rna of len M
        list_of_boxes (list): list of the bounding boxes in this matrix
        crop_bbox (list): if provided, it plots a black rectangle
        list_of_predictions (list): if provided, it plots the predicted the bounding boxes 
    Returns:
        plot: matplotilb plot of the hydrogen bond matrix with all the bboxes 
    """
        
    plt.rcParams["figure.figsize"] = (int(len([b for b in cdna1])/5), int(len([b for b in cdna2])/5))

    plt.imshow(matrix.T, cmap = cmap);

    plt.xticks(range(len([b for b in cdna1])), [b for b in cdna1], size='small')
    plt.yticks(range(len([b for b in cdna2])), [b for b in cdna2], size='small')
    ax = plt.gca()
    colors = itertools.cycle('ygrcmk')
    
    for i, (x, y, w, h) in enumerate(list_of_boxes):
        ax.add_patch(plt.Rectangle((x, y), w-1, h-1, fill=False, color=next(colors), linewidth=4))
    
    if len(crop_bbox) == 4:
        X1, Y1, W_crop, H_crop = crop_bbox
        ax.add_patch(plt.Rectangle((X1, Y1), W_crop-1, H_crop-1, fill=False, color='black', linestyle = '--', linewidth = 7))
            
    plt.colorbar()
    plt.show()

"""
-------------- L I M E -------------- L I M E -------------- L I M E --------------
"""

    
def expl_matrix_treshold(expl_matrix, treshold = 99, normalize = True):
    critical = np.percentile(expl_matrix, q = treshold)
    expl_matrix_tr = expl_matrix.copy()
    expl_matrix_tr[np.where(expl_matrix<critical)] = 0
    if normalize:
        expl_matrix_tr=(expl_matrix_tr-expl_matrix_tr.min())/(expl_matrix_tr.max()-expl_matrix_tr.min()) # It doesnt change the cosine similarity
    return expl_matrix_tr


def cosine_similarity_expl(expl_matrix, bbox_coord):
    x1, x2, y1, y2 = bbox_coord
    real_explainability = np.zeros(expl_matrix.shape)
    real_explainability[x1:x2, y1:y2] = 1
    cos_sim = F.cosine_similarity(torch.tensor(expl_matrix.flatten()), 
                                  torch.tensor(real_explainability.flatten()), 
                                  dim = 0)
    return cos_sim

def estimate_bbox(expl_matrix_tr, desired_dim = 35):
    #convolution-based approach
    """
    expl_matrix_tr: lime matrix
    desired_dim: width and height of the bounding box
    """
    
    stride = desired_dim


    output_shape1 = int(expl_matrix_tr.shape[0]/desired_dim)
    output_shape2 = int(expl_matrix_tr.shape[1]/desired_dim)

    k1 =  int(expl_matrix_tr.shape[0] - (output_shape1 - 1)*stride - 1)
    k2 =  int(expl_matrix_tr.shape[1] - (output_shape2 - 1)*stride - 1)

    A = Variable(torch.tensor(expl_matrix_tr).unsqueeze(0).unsqueeze(0).type(torch.double))
    M = Variable(torch.ones(1, 1, k1, k2).type(torch.double))
    output = F.conv2d(A, M, stride = stride).squeeze()
    if len(output.shape) == 1:
        output = output.unsqueeze(-1)

    # a questo punto prendiamo le coordinate della bbox associata alla massima cella. 
    bbox_dim = np.array(expl_matrix_tr.shape)/output.shape #width and height of each bbox
    idx_max = np.unravel_index(output.argmax(), output.shape) #coords of the output cell where we have the maximium variation
    x1hat, y1hat = (bbox_dim * (idx_max[0], idx_max[1])).astype(int)
    x2hat, y2hat = (bbox_dim * (idx_max[0] + 1, idx_max[1] + 1)).astype(int) #add + 1 because the python index starts with 0
    what, hhat = x2hat-x1hat, y2hat-y1hat

    return x1hat, y1hat, what, hhat