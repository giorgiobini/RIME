import kipoiseq
import numpy as np
import torch
from transformers import BertModel, BertConfig, DNATokenizer
import sys
import os


#from UFold_dependencies/ufold/utils.py
ss_dict = { '.': 0,  '(': 1,  ')': 2}
standard_tokenizer = DNATokenizer.from_pretrained('dna6')

def dot2onehot(dot):
    """
    Args:
        dot bracket sequence (str): ex. ...)..()..
    Returns:
        torch one hot encoded tensor (torch.tensor): One hot encoding vector of shape (N * 3)
    """
    dot = torch.tensor([ss_dict[i] for i in dot])
    return torch.nn.functional.one_hot(dot, num_classes=3).to(torch.float32)

def one_hot_encode(sequence):
    """
    Args:
        DNA sequence (str): cDNA sequence of length N
    Returns:
        one-hot vector (np.array): One hot encoding vector of shape (N * 4)
    """
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)



class DNABERTEmbedder:
    """
    6-mer sequence embedder (the input is transformed in 6-mer sequences). Credits: https://github.com/jerryji1993/DNABERT
    Args:
        dir_to_pretrained_model: path to the pre-trained model directory
    """
    def __init__(self, dir_to_pretrained_model):
        self.config = BertConfig.from_pretrained('https://raw.githubusercontent.com/jerryji1993/DNABERT/master/src/transformers/dnabert-config/bert-config-6/config.json')
        self.tokenizer = standard_tokenizer
        self.model = BertModel.from_pretrained(dir_to_pretrained_model, config=self.config)
        
    def extract_pretrained_embeddings(self, sequence):
        """
        Credits: https://github.com/jerryji1993/DNABERT/issues/11
        Args:
            DNA sequence (str): cDNA sequence of length N
        Returns:
            numpy embedding vector (np.array): vector of shape ( (N-5) * 768 ) 
        """
        model_input = prepare_sequence_for_DNABERT(sequence, self.tokenizer, k = 6)
        model_input = torch.tensor(model_input, dtype=torch.long)
        model_input = model_input.unsqueeze(0)   #model_input.shape is (N-5+2) (the +2 stands for CLS and SEP chars); unsqueeze(0) to generate a fake batch with batch size one

        output = self.model(model_input) 
        # output[0] is the embedding of all the k-mers and has dimension ((N-5+2)*768) (the +2 stands for CLS and SEP chars); 
        # output[1] is the embedding of the input sequence.
        embedded_seqs = output[0].squeeze().detach().numpy()[1:-1,] #exclude the first CLS and the last SEP chars from the
        
        try:
            assert embedded_seqs.shape[0] == (len(sequence) - 5)
        except:
            raise ValueError("The embedding shape is {} instead of {}".format(embedded_seqs.shape[0], (len(sequence) - 5)))
        return embedded_seqs

def prepare_sequence_for_DNABERT(sequence, tokenizer = standard_tokenizer, k = 6):
    kmer_seq = build_kmers(sequence, k = k)
    model_input = tokenizer.encode_plus(kmer_seq, add_special_tokens=True, max_length=512)["input_ids"]
    return model_input
    
    
def build_kmers(sequence, k = 6):
    """
    e.g. input:  "AATCTAGCA", a string of length 9
        output: "AATCTA ATCTAG TCTAGC CTAGCA" a string of 4 6-mers divided by a space char
    """
    kmers = ''
    n_kmers = len(sequence) - k + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + k]
        kmers += kmer + ' '

    return kmers[:-1] #remove last space