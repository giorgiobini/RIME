import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

# from FCN import FCNNet
from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset, RNASSDataGenerator_input, Generator_single_input
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
#from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections
import subprocess

args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess

def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis = 1).clamp_max(1))
    seq[contact.sum(axis = 1) == 0] = -1
    return seq

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def get_ct_dict(predict_matrix,batch_num,ct_dict):
    
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
'''
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in torch.nonzero(seq_embedding,as_tuple=False)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict
# randomly select one sample from the test set and perform the evaluation
'''
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    #pdb.set_trace()
    #print(seq_name)
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmpp = np.copy(seq_tmp)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    letter='AUCG'
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    #seq = ((seq_tmp+1).squeeze()[:len(seq_letter)],torch.arange(predict_matrix.shape[-1]).numpy()[:len(seq_letter)]+1)
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    dot_file_dict[batch_num] = [(seq_name.replace('/','_'),seq_letter,dot_list[:len(seq_letter)])]
    #pdb.set_trace()
    ct_file_output(ct_dict[batch_num],seq_letter,seq_name,'results/save_ct_file')
    _,_,noncanonical_pairs = type_pairs(ct_dict[batch_num],seq_letter)
    tertiary_bp = [list(x) for x in set(tuple(x) for x in noncanonical_pairs)]
    str_tertiary = []
    for i,I in enumerate(tertiary_bp):
        if i==0:
            str_tertiary += ('(' + str(I[0]) + ',' + str(I[1]) + '):color=""#FFFF00""')
        else:
            str_tertiary += (';(' + str(I[0]) + ',' + str(I[1]) + '):color=""#FFFF00""')

    tertiary_bp = ''.join(str_tertiary)
    #return ct_dict,dot_file_dict
    return ct_dict,dot_file_dict,tertiary_bp

def ct_file_output(pairs, seq, seq_name, save_result_path):

    #pdb.set_trace()
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0]-1] = int(I[1])
        #col5[I[1]] = int(I[0]) + 1
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    #os.chdir(save_result_path)
    #print(os.path.join(save_result_path, str(id[0:-1]))+'.spotrna')
    np.savetxt(os.path.join(save_result_path, seq_name.replace('/','_'))+'.ct', (temp), delimiter='\t', fmt="%s", header='>seq length: ' + str(len(seq)) + '\t seq name: ' + seq_name.replace('/','_') , comments='')

    return

def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]
    # seq_pairs = [[sequence[i[0]],sequence[i[1]]] for i in pairs]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0]-1],sequence[i[1]-1]] in [["A","U"], ["U","A"]]:
            AU_pair.append(i)
        elif [sequence[i[0]-1],sequence[i[1]-1]] in [["G","C"], ["C","G"]]:
            GC_pair.append(i)
        elif [sequence[i[0]-1],sequence[i[1]-1]] in [["G","U"], ["U","G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
        # print(watson_pairs_t, wobble_pairs_t, other_pairs_t)
    return watson_pairs_t, wobble_pairs_t, other_pairs_t


class UFoldModel:
    def __init__(self, model_device, MODEL_SAVED, eval_mode = True):
        self.device = torch.device(model_device)
        contact_net = FCNNet(img_ch=17)
        print('==========Start Loading Pretrained Model==========')
        contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location=model_device))
        print('==========Finish Loading Pretrained Model==========')
        contact_net.to(self.device)
        if eval_mode:
            self.contact_net = contact_net.eval()
        else:
            self.contact_net = contact_net
    def contact_net_prediction(self, seq_embeddings, seq_lens, seq_ori, seq_name, postprocessing = True):
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(self.device)
        with torch.no_grad():
            pred_contacts = self.contact_net(seq_embedding_batch)
            #THIS IS WHAT THEY DO IN TRAINING, THEY TRAIN IN A SUPERVISED SETTING OVER pred_contacts AFTER THIS MASKING
            #contact_masks = torch.zeros_like(pred_contacts)
            #contact_masks[:, :seq_lens, :seq_lens] = 1
            #pred_contacts = pred_contacts*contact_masks
            
        if postprocessing:
            seq_ori = torch.Tensor(seq_ori.float()).to(self.device)
            u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
            return u_no_train[:, :seq_lens, :seq_lens]
        else:
            return pred_contacts[:, :seq_lens, :seq_lens]
        
    def contact_net_output(self, seq_embeddings, seq_ori, postprocessing = True):
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(self.device)
        with torch.no_grad():
            pred_contacts = self.contact_net(seq_embedding_batch)
            
        if postprocessing:
            seq_ori = torch.Tensor(seq_ori.float()).to(self.device)
            u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5)
            return u_no_train
        else:
            return pred_contacts
        
def from_matrix_to_dot(predict_matrix):
    #this is not written in pytorch
    predict_matrix = (predict_matrix > 0.5).float()
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    dot_list = seq2dot((seq_tmp+1).squeeze())
    return dot_list
        
def from_one_hot_to_dot(one_hot):
    seq = one_hot.argmax(-1).cpu().numpy()
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq == 0] = '('
    dot_file[seq == 1] = ')'
    dot_file[seq == 2] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def from_matrix_to_one_hot(predict_matrix, device = 'cuda:0'):
    #pytorch version of from_matrix_to_dot function
    ## input: torch.Size([batch, m, m])
    ## output: torch.Size([batch, m, 3])
    predict_matrix = (predict_matrix > 0.5).float()
    seq_tmp = torch.mul(predict_matrix.argmax(axis=1), predict_matrix.sum(axis = 1).clamp_max(1))
    seq_tmp[predict_matrix.sum(axis = 1) == 0] = -1
    seq = (seq_tmp+1)
    idx = torch.arange(1, seq.shape[-1] + 1, device = device)
    dot_file = torch.zeros(seq.shape, device = device, dtype=int)
    dot_file[seq < idx] = 1
    dot_file[seq == 0] = 2
    return torch.nn.functional.one_hot(dot_file).to(dtype=torch.float32)
        
def get_seq_embeddings_seq_ori(sequence, name):
    test_data = Generator_single_input(sequence, name)
    test_set = Dataset_FCN(test_data)
    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 1,'drop_last': False}
    test_generator = data.DataLoader(test_set, **params) 
    for batch, (seq_embeddings, seq_lens, seq_ori, seq_name) in enumerate(test_generator):
        break
    return seq_embeddings, seq_lens, seq_ori, seq_name