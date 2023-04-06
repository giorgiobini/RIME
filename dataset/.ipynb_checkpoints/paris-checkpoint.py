import json
import time
import os
import numpy as np
from typing import Any, Tuple, List
from pathlib import Path
import torch
import pandas as pd

class PARIS:
    def __init__(self, pairs_folder, annotation_file, embedding_is_already_first = False, create_fake_dim = True):
        """ 
        I took inspiration from the COCO class
        Constructor of the PARIS class for reading and visualizing annotations.
        :param pairs_folder (str): location to the folder that hosts the rna embeddings.
        :param annotation_file (str): location of annotation file
        :embedding_is_already_first (bool): let d indicate the dimension of the embeddings and 
                                            L the length of the rna.
                                            If the data are stored such that the shape of the rna 
                                            is (d * N), then set embedding_is_already_first = True, 
                                            otherwise keep it as False.
        :create_fake_dim (bool): if True, rna will be reshaped with a new dimension. 
                                 e.g. input shape is (768 * 10), output shape is (768 * 10 * 1)                             
        :return:
        """
        
        self.pairs_folder = pairs_folder
        self.dataset = dict()
        self.embedding_is_already_first = embedding_is_already_first
        self.create_fake_dim = create_fake_dim
        assert annotation_file is not None
        print('loading annotations into memory...')
        tic = time.time()
        dataset = json.load(open(annotation_file, 'r'))
        assert type(dataset)==list, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        self.dataset = dataset
        self.createIndex()
        

    def createIndex(self):
        pairs = {}
        # create index
        for pair in self.dataset:
            pairs[pair['id']] = pair #I filter based on what I need
        print('index created!')

        # create class member
        self.pairs = pairs
        
        #create ids
        self.ids = list(sorted(pairs.keys()))
    
    def get_dataframe(self, return_cdna = False):
        #it will return a pandas DataFrame with some statistics
        df = pd.DataFrame.from_dict(self.pairs, orient = 'index').reset_index(drop = True) # columns : ['id', 'rna_rna_pair', 'file_name', 'specie', 'gene1', 'gene2', 'there_is_interaction', 'rna1_length', 'rna2_length', 'boxes', 'relative_boxes', 'id', 'cdna1', 'cdna2']
        columns_to_keep = ['id', 'specie', 'gene1', 'gene2', 'there_is_interaction', 'rna1_length', 'rna2_length', 'boxes']
        if return_cdna: 
            columns_to_keep.append('cdna1')
            columns_to_keep.append('cdna2')
        df = df.filter(columns_to_keep, axis = 1)
        df['n_interactions'] = df.apply(lambda x: len(x['boxes']), axis = 1)
        df = df.drop('boxes', axis = 1)
        return df
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def prepare(self, rna):
        if (self.embedding_is_already_first == False):
            #embedding must be first in the dimension
            rna = rna.permute(1,0)
            
        if self.create_fake_dim:
            #if you use the nested_tensor_from_tensor_list (see util/misc), you need 3 dimensions, so it's better to use this trick
            rna = rna.unsqueeze(-1)
        return rna
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id_ = self.ids[index]
        pair = self.pairs[id_]
        rna1, rna2 = np.load(os.path.join(self.pairs_folder, pair["file_name"]), allow_pickle=True)
        rna1, rna2 = torch.tensor(rna1), torch.tensor(rna2)
        rna1, rna2 = self.prepare(rna1), self.prepare(rna2)
        if (pair['there_is_interaction'] == True):
            relative_boxes = torch.tensor(pair['relative_boxes'])
            labels = torch.ones(relative_boxes.shape[0], dtype = torch.int64)
        else:
            relative_boxes = torch.empty((0,4), dtype = torch.float32)
            labels = torch.empty((0),dtype = torch.int64)
            
        rnas = (rna1, rna2)
        target = {'pairs_id': torch.tensor([int(pair['id'])]), 
                  'labels': labels, 
                  'boxes':relative_boxes,
                  'orig_size':torch.as_tensor([int(pair['rna1_length']), int(pair['rna2_length'])])
                 }
        
        return rnas, target

    
def build(pairs_set, args, load_cdna = False):
    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    PATHS = {
        "train": (root / "data_folder", root / "annotation_files_folder" / 'training_annotations.json'), #training_annotations validation_annotations test_annotations
        "val": (root / "data_folder", root / "annotation_files_folder" / 'test_annotations.json'),
    }

    pairs_folder, ann_file = PATHS[pairs_set]
    dataset = PARIS(pairs_folder, ann_file)
    return dataset