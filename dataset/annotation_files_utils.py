import sys
sys.path.insert(0, '..')
from util import contact_matrix
from dataset import train_test_val_utils
import torch
import numpy as np
import os
import pandas as pd
import string
import random
from util import box_ops
import json
import itertools
from ortools.sat.python import cp_model
from copy import copy

def random_dna_generator(size):
    return ''.join(random.choice('AGCT') for _ in range(size))

def obtain_minx1y1_maxx2y2(list_of_boxes):
    """
    Function for the linear programming problem. It provides the variables min_x1, min_y1, max_x2, max_y2 (see explaination in the Create Annotation Files.ipynb notebook.
    
    Args:
        list_of_bboxes (list): list of the bounding boxes in the contact matrix
    Returns:
        min_x1 (int): The minimum coordinate of x1 between all the x1_i values (where i = 1, ... len(list_of_boxes))
        min_y1 (int): The minimum coordinate of y1 between all the y1_i values (where i = 1, ... len(list_of_boxes))
        max_x2 (int): The maximum coordinate of x2 between all the x2_i values (where i = 1, ... len(list_of_boxes))
        max_y2 (int): The maximum coordinate of y2 between all the y2_i values (where i = 1, ... len(list_of_boxes))
    """
    min_x1 = float('inf')
    min_y1 = float('inf')
    max_x2 = 0
    max_y2 = 0 

    for x1, y1, w, h in list_of_boxes:
        x2, y2 = x1 + w, y1 + h

        if x1 < min_x1:
            min_x1 = x1
        if y1 < min_y1:
            min_y1 = y1

        if x2 > max_x2:
            max_x2 = x2
        if y2 > max_y2:
            max_y2 = y2
            
    return min_x1, min_y1, max_x2, max_y2

class LpProblem:
    """
    Linear programming problem (see explaination in the Create Annotation Files.ipynb notebook.
    
     Args:
        min_rna_length (int): minimum length admitted for the length of a sampled rna.
        max_rna_length (int): maximum length admitted for the length of a sampled rna.
        min_x1 (int): The minimum coordinate of x1 between all the x1_i values (where i = 1, ..., len(list_of_boxes))
        min_y1 (int): The minimum coordinate of y1 between all the y1_i values (where i = 1, ..., len(list_of_boxes))
        max_x2 (int): The maximum coordinate of x2 between all the x2_i values (where i = 1, ..., len(list_of_boxes))
        max_y2 (int): The maximum coordinate of y2 between all the y2_i values (where i = 1, ..., len(list_of_boxes))
        N (int): the rna1 length 
        M (int): the rna2 length 
        solution (str): it can be random, min, or max
    Returns:
        crop_bounding_box (list): a bounding box with format x1y1wh, which represents the coordinates of the cropped (sampled) contact matrix. The sampled contact matrix is such that all the interaction bounding boxes are included (if there are interactions in the rna rna pair)
    """
    def __init__(self, min_rna_length, max_rna_length, min_x1, min_y1, max_x2, max_y2, N, M, solution):
        self.min_rna_length = min_rna_length
        self.max_rna_length = max_rna_length
        self.min_x1 = min_x1
        self.min_y1 = min_y1
        self.max_x2 = max_x2
        self.max_y2 = max_y2
        self.N = N
        self.M = M
        self.solution = solution
        self.success, self.res_maximize_L = get_solution_max_min(min_rna_length, max_rna_length, min_x1, min_y1, max_x2, max_y2, N, M, get_min = False)    
        
    def get_crops_(self, n_crops):
        """
        Obtain one sample from the rna_rna pair.
        - (sampling_criteria['min_rna_length'] < len(rna1_sample) < sampling_criteria['max_rna_length'])&(sampling_criteria['min_rna_length'] < len(rna1_sample) < sampling_criteria['max_rna_length'])
        - if the original rna_rna pair has I interactions, then I interactions must be included in the sample.
        - if an interaction is included, then it have to be included entirely (it cannot be cutted).
        """
        assert self.success == True
        X1_max, Y1_max, X2_max, Y2_max = self.res_maximize_L       
        max_objective_function_value = - X1_max - Y1_max + X2_max + Y2_max
        crops_max = get_solutions_maximizing_minimizing(self.min_rna_length, self.max_rna_length, self.min_x1, self.min_y1, self.max_x2, self.max_y2, self.N, self.M, max_min_objective_function_value = max_objective_function_value, n_samples = n_crops)
        if self.solution == 'max':
            crops = crops_max
            
        elif self.solution == 'random':
            self.success, self.res_minimize_L = get_solution_max_min(self.min_rna_length, self.max_rna_length, self.min_x1, self.min_y1, self.max_x2, self.max_y2, self.N, self.M, get_min = True)    
            X1_min, Y1_min, X2_min, Y2_min = self.res_minimize_L
            min_objective_function_value = - X1_min - Y1_min + X2_min + Y2_min
            crops_min = get_solutions_maximizing_minimizing(self.min_rna_length, self.max_rna_length, self.min_x1, self.min_y1, self.max_x2, self.max_y2, self.N, self.M, max_min_objective_function_value = min_objective_function_value, n_samples = n_crops)
            if (len(crops_min) < len(crops_max)):
                crops_min = random.choices(crops_min, k = len(crops_max)) #oversample crops_min
            elif (len(crops_min) > len(crops_max)):
                crops_max = random.choices(crops_max, k = len(crops_min)) #oversample crops_max
            assert len(crops_min) == len(crops_max)
                
            crops = []
            for i in range(len(crops_min)):
                X1_max, Y1_max, X2_max, Y2_max = crops_max[i]
                X1_min, Y1_min, X2_min, Y2_min = crops_min[i]
                X1 = np.random.randint(X1_max, X1_min+1)
                Y1 = np.random.randint(Y1_max, Y1_min+1)
                X2 = np.random.randint(X2_min, X2_max+1)
                Y2 = np.random.randint(Y2_min, Y2_max+1)
                crops.append([X1, Y1, X2, Y2])
                
        elif self.solution == 'min':
            self.success, self.res_minimize_L = get_solution_max_min(self.min_rna_length, self.max_rna_length, self.min_x1, self.min_y1, self.max_x2, self.max_y2, self.N, self.M, get_min = True)    
            X1_min, Y1_min, X2_min, Y2_min = self.res_minimize_L
            min_objective_function_value = - X1_min - Y1_min + X2_min + Y2_min
            crops = get_solutions_maximizing_minimizing(self.min_rna_length, self.max_rna_length, self.min_x1, self.min_y1, self.max_x2, self.max_y2, self.N, self.M, max_min_objective_function_value = min_objective_function_value ,n_samples = n_crops)
            
        crops = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in crops]
        
        return crops

class SolutionManager(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__best_solutions = []

    def on_solution_callback(self):
        self.__solution_count += 1
        self.__best_solutions.append([self.Value(v) for v in self.__variables])

    def solution_count(self):
        return self.__solution_count

    def best_solutions(self):
        return self.__best_solutions
    
def get_solution_max_min(min_rna_length, max_rna_length, min_x1, min_y1, max_x2, max_y2, N, M, get_min = False):
    model = cp_model.CpModel()
    # Creates the variables.
    x1 = model.NewIntVar(0, min_x1 - 1, 'x1') # 0<=𝑋1<=(min_x1 -1); min_x1 is the minimum value of x1 in all the bounding boxes of this rna_rna pair. 
    y1 = model.NewIntVar(0, min_y1 - 1, 'y1') #0<=𝑌1<=(min_y1 -1); min_y1 is the minimum value of y1 in all the bounding boxes of this rna_rna pair. 
    x2 = model.NewIntVar(max_x2 + 1, N, 'x2') #(max_x2 + 1)<=𝑋2<=N; N is the length of rna1 
    y2 = model.NewIntVar(max_y2 + 1, M, 'y2') #(max_y2 + 1)<=𝑌2<=M; M is the length of rna2
    
    # Creates the constraints.
    model.Add(x2 - x1 < max_rna_length) #(𝑋2−𝑋1)<sampling_criteria['max_rna_length'] 
    model.Add(x2 - x1 > min_rna_length) #(𝑋2−𝑋1)>sampling_criteria['min_rna_length'] 
    model.Add(y2 - y1 < max_rna_length) #(𝑌2−𝑌1)<sampling_criteria['max_rna_length'] 
    model.Add(y2 - y1 > min_rna_length) #(𝑌2−𝑌1)>sampling_criteria['min_rna_length']
    if get_min:
        model.Minimize(- x1 - y1 + x2 + y2)
    else:
        model.Maximize(- x1 - y1 + x2 + y2)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if (status == cp_model.OPTIMAL) or (status == cp_model.FEASIBLE):
        success = True
        best_x1 = solver.Value(x1)
        best_y1 = solver.Value(y1)
        best_x2 = solver.Value(x2)
        best_y2 = solver.Value(y2)
        res_maximize_or_minimize_L = [best_x1, best_y1, best_x2, best_y2]
    else:
        success = False 
        res_maximize_or_minimize_L = []
    return success, res_maximize_or_minimize_L

def get_solutions_maximizing_minimizing(min_rna_length, max_rna_length, min_x1, min_y1, max_x2, max_y2, N, M, max_min_objective_function_value, n_samples):
    #find the optimal solution that maximize the objective (-x1 - y1 + x2 + y2)
    model = cp_model.CpModel()
    # Creates the variables.
    x1 = model.NewIntVar(0, min_x1 - 1, 'x1') # 0<=𝑋1<=(min_x1 -1); min_x1 is the minimum value of x1 in all the bounding boxes of this rna_rna pair. 
    y1 = model.NewIntVar(0, min_y1 - 1, 'y1') #0<=𝑌1<=(min_y1 -1); min_y1 is the minimum value of y1 in all the bounding boxes of this rna_rna pair. 
    x2 = model.NewIntVar(max_x2 + 1, N, 'x2') #(max_x2 + 1)<=𝑋2<=N; N is the length of rna1 
    y2 = model.NewIntVar(max_y2 + 1, M, 'y2') #(max_y2 + 1)<=𝑌2<=M; M is the length of rna2
    
    # Creates the constraints.
    model.Add(x2 - x1 < max_rna_length) #(𝑋2−𝑋1)<sampling_criteria['max_rna_length'] 
    model.Add(x2 - x1 > min_rna_length) #(𝑋2−𝑋1)>sampling_criteria['min_rna_length'] 
    model.Add(y2 - y1 < max_rna_length) #(𝑌2−𝑌1)<sampling_criteria['max_rna_length'] 
    model.Add(y2 - y1 > min_rna_length) #(𝑌2−𝑌1)>sampling_criteria['min_rna_length']
    
    model.Add(- x1 - y1 + x2 + y2 == max_min_objective_function_value)    
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 7.0
    solver.parameters.enumerate_all_solutions = True
    solution_manager = SolutionManager([x1, y1, x2, y2])
    status = solver.Solve(model, solution_manager)
    assert status == cp_model.OPTIMAL or status == cp_model.FEASIBLE
    best_solutions = solution_manager.best_solutions()
    n_solutions = solution_manager.solution_count()
    assert n_solutions == len(best_solutions)

    if n_solutions > n_samples:
        best_solutions = random.sample(best_solutions, n_samples)

    return best_solutions

class GenePairManager:
    """
    This class allows to deal with a single rna_rna pair that has zero, one, or more bboxes. I will use the following notation:
        - rna_1 and gene_1 are synonyms (rna_2 and gene_2 as well)
        - N is the rna1 length 
        - M is the rna2 length
    """
    def __init__(self, gene1, gene2, cdna1, cdna2, list_of_boxes, sampling_criteria, interactive_pair, solution = 'random'):
        self.gene1 = gene1
        self.gene2 = gene2
        self.cdna1 = cdna1
        self.cdna2 = cdna2
        self.sampling_criteria = sampling_criteria
        self.list_of_boxes = list_of_boxes
        self.interactive_pair = interactive_pair
        self.N = len(cdna1)
        self.M = len(cdna2)
        self.list_of_relative_boxes = [box_ops.from_original_to_relative_boxes(boxes, (self.N, self.M))[0]
                                       for boxes in list_of_boxes]   
        assert solution in ['random', 'max', 'min']
        self.solution = solution
        
        if self.interactive_pair:
            min_x1, min_y1, max_x2, max_y2 = obtain_minx1y1_maxx2y2(self.list_of_boxes)
        else:
            min_x1, min_y1, max_x2, max_y2 = self.N, self.M, 0, 0 # values I want in order to obtain a random crop
            
        lp_problem = LpProblem(int(self.sampling_criteria['min_rna_length']),
                               int(self.sampling_criteria['max_rna_length'] + 1), 
                               int(min_x1),
                               int(min_y1),
                               int(max_x2),
                               int(max_y2),
                               int(self.N),
                               int(self.M),
                               self.solution
                              )
        
        self.lp_problem = lp_problem
        self.is_possible_to_sample = lp_problem.success

    def encoded_rnas(self, encoding_function):
        return encoding_function(self.cdna1), encoding_function(self.cdna2)
    
    def one_hot_contact_matrix(self, encoding_function):
        encoded_1, encoded_2 = self.encoded_rnas(encoding_function)
        return contact_matrix.create_contact_matrix(torch.tensor(np.expand_dims(encoded_1, 0)), 
                     torch.tensor(np.expand_dims(encoded_2, 0)))
    
    def plot(self, crop_bbox = [], plot_geoms = False):
        contact_matrix.plot_contact_matrix(self.cdna1, self.cdna2, self.list_of_boxes, crop_bbox = crop_bbox, plot_geoms = plot_geoms)
        
    def clean_bounding_boxes(self):
        list_of_indexes = []
        list_bboxes = self.list_of_boxes
        for idx in range(len(list_bboxes)):
            list_of_indexes.append(IndexNode(idx))
        index_pairs = list(itertools.combinations(list(range(len(list_bboxes))), 2))
        for index_pair in index_pairs:
            if box_ops.bboxes_overlaps(list_bboxes[index_pair[0]], list_bboxes[index_pair[1]]):
                list_of_indexes[index_pair[0]].add_link(list_of_indexes[index_pair[1]])
        nodes = set(list_of_indexes)
        
        # Find all the connected components.
        list_group_indexes = []
        for components in connected_components(nodes):
            group_indexes = sorted(node.name for node in components)
            list_group_indexes.append(group_indexes)
        new_list_of_boxes = []
        for group_idx in list_group_indexes:
            if len(group_idx)>0:

                group = np.array(list_bboxes)[group_idx,] 
                #now it's x1, y1, w, h

                group[:,2] = group[:,0] + group[:,2]
                group[:,3] = group[:,1] + group[:,3]
                #now it's x1, y1, x2, y2

                min_x1 = group.min(axis = 0)[0]
                min_y1 = group.min(axis = 0)[1]
                max_x2 =  group.max(axis = 0)[2]
                max_y2 = group.max(axis = 0)[3]

                w, h = (max_x2-min_x1), (max_y2-min_y1)

                new_list_of_boxes.append([min_x1, min_y1, w, h])
            else:
                new_list_of_boxes.append(list_bboxes[group_idx[0]])
        self.list_of_boxes = new_list_of_boxes
    
    def get_crops(self, n_crops):
        assert self.is_possible_to_sample
        crops = self.lp_problem.get_crops_(n_crops)
        return crops
            
    def get_sample(self, crop_bounding_box):
        X1, Y1, W_crop, H_crop = crop_bounding_box
        X2 = X1 + W_crop
        Y2 = Y1 + H_crop
        new_cdna1 = self.cdna1[int(X1):int(X2)]
        new_cdna2 = self.cdna2[int(Y1):int(Y2)]
        new_list_of_boxes = []
        for old_x1, old_y1, w, h in self.list_of_boxes:
            x1 = old_x1 - X1
            y1 = old_y1 - Y1
            new_list_of_boxes.append([x1, y1, w, h])
        return GenePairManager(self.gene1, self.gene2, new_cdna1, new_cdna2, new_list_of_boxes, self.sampling_criteria, self.interactive_pair, self.solution)
        
class GenePairSampler:
    def __init__(self, df, df_cdna, sampling_criteria, solution):
        self.df = df
        self.df_cdna = df_cdna
        self.sampling_criteria = sampling_criteria
        self.solution = solution
        
        
    def get_pair(self, couple, swap_axes = False):
        subset = self.df[self.df.couples == couple]
        len(subset.there_is_interaction.unique()) == 1
        interactive_pair = subset.iloc[0].there_is_interaction
        gene1, gene2 = couple.split('_')
        cdna1 = self.df_cdna[self.df_cdna.gene_id == gene1].cdna.iloc[0]
        cdna2 = self.df_cdna[self.df_cdna.gene_id == gene2].cdna.iloc[0]
        
        if interactive_pair:
            list_of_boxes = subset.filter(['x1', 'y1', 'w', 'h']).values.tolist()
        else:
            list_of_boxes = []
            
        if swap_axes:
            gene1, gene2 = gene2, gene1
            cdna1, cdna2 = cdna2, cdna1
            if len(list_of_boxes)>0:
                new_list_of_boxes = []
                for b in list_of_boxes:
                    x1, y1, w, h = b
                    new_list_of_boxes.append([y1, x1, h, w])
                list_of_boxes = new_list_of_boxes

        return GenePairManager(gene1, gene2, cdna1, cdna2, list_of_boxes, self.sampling_criteria, interactive_pair, self.solution) #sampling_criteria is a dictionay
    
def create_boxes_xywh(row):
    """
    The reason why I am taking the max in this function max((row.start_map1 - 1), 0) is that:
        start_map1, end_map1, start_map2, end_map2 are in the interval [1, len(rna)] for the interactive pairs (indeed, the interactive regions must be sliced like: cdna1[(start_map1-1):end_map1])
        start_map1, end_map1, start_map2, end_map2 are 0,0,0,0 for the non interactive pairs
    
    Args:
        a row (pd.Series) of the dataset.
    Returns:
        boxes (list): A list of bboxes  with the form -> bbox = [x, y, w, h] 
    """
    assert (row.gene1 == row.gene_id1)&(row.gene2 == row.gene_id2)
    x = max((row.start_map1 - 1), 0)
    y = max((row.start_map2 - 1), 0)
    w = row.end_map1 - max(0, (row.start_map1 - 1))
    h = row.end_map2 - max(0, (row.start_map2 - 1))
    assert row.area_of_the_interaction == h*w
    return pd.Series([x, y, w, h])

def get_df_cdna(original_files_data_dir, processed_files_data_dir):
    """
    Since all the genes in the non-interaction file are contained in the genes in the interaction file, I can use only df_int in order to create this dataframe.
    """
    filename = os.path.join(processed_files_data_dir, 'df_cdna.csv')
    if os.path.isfile(filename):
        df_cdna = pd.read_csv(filename)
    else:
        df_int = train_test_val_utils.read_dataframe(os.path.join(original_files_data_dir, 'rise_paris_tr.new.mapped_interactions.tx_regions.txt'), columns_to_drop = ['Unnamed: 0'])
        df_int = df_int.filter(['gene_id1',
                                'gene_id2', 
                                'cdna_1', 
                                'cdna_2', 
                                'species',
                                'length_1', 
                                'length_2',
                                'ensembl_transcript_id_1',
                                'ensembl_transcript_id_2'
                               ], axis =1)
        df_int1 = df_int[['gene_id1', 
                          'cdna_1', 
                          'species', 
                          'length_1',
                          'ensembl_transcript_id_1']].rename({'gene_id1':'gene_id',
                                                              'cdna_1':'cdna',
                                                              'length_1':'length', 
                                                              'ensembl_transcript_id_1': 'ensembl_transcript_id'},
                                                             axis =1)
        df_int2 = df_int[['gene_id2', 
                          'cdna_2', 
                          'species', 
                          'length_2',
                          'ensembl_transcript_id_2']].rename({'gene_id2':'gene_id',
                                                              'cdna_2':'cdna', 
                                                              'length_2':'length',
                                                              'ensembl_transcript_id_2':'ensembl_transcript_id'},
                                                             axis =1)
        df_cdna = pd.concat([df_int1, df_int2], axis = 0).reset_index(drop = True)
        assert df_cdna.drop_duplicates().shape[0] == len(df_cdna.gene_id.unique())
        df_cdna = df_cdna.drop_duplicates()
        
        tr_anns = pd.read_csv(os.path.join(original_files_data_dir, 'tx_regions.ens84.txt'), sep = '\t').drop(['Unnamed: 0', 'length'], axis = 1)
                
        assert df_cdna.merge(tr_anns, on = ['ensembl_transcript_id'], how = 'inner').shape[0] == df_cdna.shape[0]

        df_cdna = df_cdna.merge(tr_anns, on = ['ensembl_transcript_id'], how = 'inner')
        df_cdna = df_cdna.drop('ensembl_transcript_id', axis = 1)
        df_cdna.loc[df_cdna["UTR5"] == '/', ["UTR5", "CDS", "UTR3"]] = -1 #need all numeric cols 
        df_cdna['UTR5_start'] = 0
        df_cdna['UTR5_end'] = df_cdna.UTR5.astype('int')
        df_cdna['CDS_end'] = df_cdna.CDS.astype('int')
        df_cdna['UTR3_end'] = df_cdna.UTR3.astype('int')
        df_cdna['protein_coding'] = False
        df_cdna.loc[df_cdna["UTR5"] == -1, ["UTR5_start", "UTR5_end", "CDS_end", "UTR3_end"]] = np.nan
        df_cdna.loc[df_cdna["UTR5_end"].isna() == False, "protein_coding"] = True
        df_cdna = df_cdna.drop(['UTR5', 'CDS', 'UTR3'], axis = 1)
        df_cdna.to_csv(filename, index = False)
    return df_cdna
        
def get_df(dataset_data_dir, cleaned_version = False):
    """
    Args:
        dataset_data_dir (str): directory where to load the data
        cleaned_version (bool): if True, then you load the version where bounding boxes are cleaned (where overlapping bboxes are mapped into a common big box)
    Returns:
        df (Pd.Dataframe): dataframe
    """
    if cleaned_version:
        filename = os.path.join(dataset_data_dir, 'df_annotation_files_cleaned.csv')
        df = pd.read_csv(filename)
    else:
        filename = os.path.join(dataset_data_dir, 'df_annotation_files.csv')
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
        else:
            df_old = pd.read_csv(os.path.join(dataset_data_dir, 'df_train_test_val.csv')) 
            # I have to better organize this df_old dataset for this annotation files creation task.
            df_old = df_old.filter(['couples', 'gene_id1', 'gene_id2',
                                    'start_map1', 'end_map1',
                                    'start_map2', 'end_map2',
                                    'area_of_the_matrix','species', 
                                    'there_is_interaction', 'area_of_the_interaction',
                                   'tx_id_1_localization', 'tx_id_2_localization'], axis = 1)
            df_temp = pd.concat([df_old, df_old.couples.str.split('_', expand = True)\
                                 .rename({0:'gene1', 1:'gene2'}, axis = 1)], axis = 1)
            df_boxes = df_temp.apply(create_boxes_xywh, axis = 1).rename({0: 'x1', 1: 'y1', 2:'w', 3:'h'}, axis = 1)
            df = pd.concat([df_temp, df_boxes], axis = 1).drop(['gene_id1', 'gene_id2', 'start_map1', 'end_map1', 'start_map2', 'end_map2'], axis = 1).rename({'tx_id_1_localization':'gene1_region', 'tx_id_2_localization':'gene2_region'}, axis = 1)
            
            # There are few couples where the annotation of the gene1_region is '5UTR-CDS' and 'CDS-3UTR'.
            # We need to change these cases
            df.loc[df['gene1_region'] == '5UTR-CDS', 'gene1_region'] = '5UTR'
            df.loc[df['gene1_region'] == 'CDS-3UTR', 'gene1_region'] = '3UTR'
            df.loc[df['gene2_region'] == '5UTR-CDS', 'gene2_region'] = '5UTR'
            df.loc[df['gene2_region'] == 'CDS-3UTR', 'gene2_region'] = '3UTR'
            
            # I prefer the annotation UTR3 than 3UTR
            df.loc[df['gene1_region'] == '5UTR', 'gene1_region'] = 'UTR5'
            df.loc[df['gene1_region'] == '3UTR', 'gene1_region'] = 'UTR3'
            df.loc[df['gene2_region'] == '5UTR', 'gene2_region'] = 'UTR5'
            df.loc[df['gene2_region'] == '3UTR', 'gene2_region'] = 'UTR3'
            
            df.to_csv(filename, index = False)
    return df

def filter_dataset(df, minimum_rna_length, maximum_rna_length, minimum_area_of_the_matrix, maximum_area_of_the_matrix):
    df_int = df[df.there_is_interaction == True].reset_index(drop = True)
    df_con = df[df.there_is_interaction == False].reset_index(drop = True)
    df_int = df_int[df_int.w > minimum_rna_length].reset_index(drop = True)
    df_int = df_int[df_int.w < maximum_rna_length].reset_index(drop = True)
    df_int = df_int[df_int.h > minimum_rna_length].reset_index(drop = True)
    df_int = df_int[df_int.h < maximum_rna_length].reset_index(drop = True)
    df_int = df_int[df_int.area_of_the_interaction > minimum_rna_length].reset_index(drop = True)
    df_int = df_int[df_int.area_of_the_interaction < maximum_area_of_the_matrix].reset_index(drop = True)
    df = pd.concat([df_int, df_con], axis = 0).sample(frac=1) #shuffle at the end
    return df

def load_train_test_val(df, path):
    return train_test_val_utils.create_or_load_train_test_val(df, path)

def save_data_and_collect_annotations(row, gpsampler, dna_embedder, matrixes_path, number_of_copy_per_rna_rna, swap_axes, annotation_list):
    """
    Args:
        row (pd.Series): pandas series of a single rna-rna pair.
        gpsampler (GenePairSampler) an initalized GenePairSampler object.
        dna_embedder (DNABERTEmbedder) an initialized DNABERTEmbedder object
        matrixes_path (str): where to save the tuple of embeddings (embeddings_cdna1, embeddings_cdna2).
        number_of_copy_per_rna_rna (int): how many (different) copies for this specific rna-rna pair you want to sample (if you use swap_axes you will have number_of_copy_per_rna_rna swapped and number_of_copy_per_rna_rna not swapped).
        swap_axes (str): If True, for each rna-rna pair you will obtain 2 different versions, one with orignal axes and one with swapped axes. 
        annotation_list (list): list of annotations for this rna-rna pair.
    Returns:
        annotation_list (list): updated annotation list.
    """
    rna_rna_pair = row.couples
    specie = row.species
    there_is_interaction = row.there_is_interaction
    gene1 = row.gene1
    gene2 = row.gene2

    n_samples = 0
    pair_list = [gpsampler.get_pair(rna_rna_pair, swap_axes = False)]
    if swap_axes:
        pair_list.append(gpsampler.get_pair(rna_rna_pair, swap_axes = True)) #swapped pair
            
    if pair_list[0].is_possible_to_sample:
        for pair in pair_list:
            for crop in pair.get_crops(n_crops = number_of_copy_per_rna_rna):
                sample = pair.get_sample(crop)
                N = len(sample.cdna1)
                M = len(sample.cdna2)
                boxes = sample.list_of_boxes
                relative_boxes = sample.list_of_relative_boxes
                tuple_matrixes = (dna_embedder.extract_pretrained_embeddings(sample.cdna1),
                                  dna_embedder.extract_pretrained_embeddings(sample.cdna2))
                file_name = rna_rna_pair + '_{}.npy'.format(n_samples)
                annotations = {'rna_rna_pair':rna_rna_pair,
                               'file_name':file_name,
                               'specie':specie,
                               'gene1':gene1,
                               'gene2':gene2,
                               'there_is_interaction':there_is_interaction,
                               'rna1_length':N,
                               'rna2_length':M,
                               'boxes':boxes,
                               'relative_boxes':relative_boxes,
                               'id': int(str(hash(file_name))),
                               'cdna1': sample.cdna1,
                               'cdna2': sample.cdna2
                              }
                annotation_list.append(annotations)
                np.save(os.path.join(matrixes_path, file_name), tuple_matrixes)
                n_samples += 1

    return annotation_list
            
def save_data_and_annotations(df_set, gpsampler, dna_embedder, matrixes_path, annotation_path, number_of_copy_per_rna_rna, which_set, swap_axes, its_jupyter_notebook = True):
    """
    Args:
        df_set (pd.DataFrame) pandas dataframe that can be either train, test or validation.
        gpsampler (GenePairSampler) an initalized GenePairSampler object.
        dna_embedder (DNABERTEmbedder) an initialized DNABERTEmbedder object
        matrixes_path (str): where to save the tuple of embeddings (embeddings_cdna1, embeddings_cdna2).
        annotation_path (str) where to save the annotation files.
        number_of_copy_per_rna_rna (int): how many (different) copies for this specific rna-rna pair you want to sample.
        which_set (str): specify if its training, test or validation.
        swap_axes (str): If True, for each rna-rna pair you will obtain 2 different versions, one with orignal axes and one with swapped axes. 
    """
    if its_jupyter_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    assert which_set in ['training', 'test', 'validation']
    annotations = []
    for row in tqdm(df_set.itertuples(), total=df_set.shape[0], mininterval=500):
        annotations = save_data_and_collect_annotations(row, gpsampler, dna_embedder, matrixes_path,
                                                          number_of_copy_per_rna_rna, swap_axes, annotations)
    with open(os.path.join(annotation_path, '{}_annotations.json'.format(which_set)), 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
        f.close()
        

class IndexNode(object):
    """
    credits: https://breakingcode.wordpress.com/2013/04/08/finding-connected-components-in-a-graph/
    """
    def __init__(self, name):
        self.__name  = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)
    
# The function to look for connected components.
def connected_components(nodes):

    # List of connected components found. The order is random.
    result = []

    # Make a copy of the set, so we can modify it.
    nodes = set(nodes)

    # Iterate while we still have nodes to process.
    while nodes:

        # Get a random node and remove it from the global set.
        n = nodes.pop()

        # This set will contain the next group of nodes connected to each other.
        group = {n}

        # Build a queue with this node in it.
        queue = [n]

        # Iterate the queue.
        # When it's empty, we finished visiting a group of connected nodes.
        while queue:

            # Consume the next item from the queue.
            n = queue.pop(0)

            # Fetch the neighbors.
            neighbors = n.links

            # Remove the neighbors we already visited.
            neighbors.difference_update(group)

            # Remove the remaining nodes from the global set.
            nodes.difference_update(neighbors)

            # Add them to the group of connected nodes.
            group.update(neighbors)

            # Add them to the queue, so we visit them in the next iterations.
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    # Return the list of groups.
    return result