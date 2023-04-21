import pandas as pd
import os
import numpy as np
import ast
import re
import itertools
import sys
sys.path.insert(0, '..')
from util import box_ops

def read_dataframe(file_path, columns_to_drop = []):
    cols = list(pd.read_csv(file_path, sep ='\t', nrows =1))
    # Use list comprehension to remove the unwanted column in **usecol**
    df = pd.read_csv(file_path, sep ='\t', usecols =[i for i in cols if i not in columns_to_drop])
    return df

def get_list_couples(d):
    l = []
    for pair in d:
        rgx = re.search(r"(.+)--(.+)", pair)
        first_gene, second_gene = rgx.group(1), rgx.group(2)
        couple = '_'.join(sorted([first_gene, second_gene]))
        # print(couple, couple in all_couples)
        # assert couple in all_couples
        l.append(couple)
    return l

def get_couple_id(g1, g2, needed_to_swap = False):
    l = sorted([g1, g2])
    if needed_to_swap:
        need_to_swap = False if l[0] == g1 else True
        return '_'.join(l), need_to_swap
    else:
        return '_'.join(l)

def create_pairs(x):
    """
    Pairs will be created in such a way that gene1_gene2 is equal to gene2_gene1. How? I will simply order the strings before create the couple string.
    """
    first_gene = x.gene_id1
    second_gene = x.gene_id2
    couple, need_to_swap = get_couple_id(first_gene, second_gene, needed_to_swap = True)
    return pd.Series([couple, need_to_swap])

def obtain_df_pos_controls(cc):
    diz = {}
    idx = 0
    for _, row in cc.iterrows():
        controlled_gene = row.controlled # controlled è il positivo
        control_gene = row.controls
        d_neg = ast.literal_eval(row.couples_negative)
        d_pos = ast.literal_eval(row.couples_rr_ctrlled)
        negatives = get_list_couples(d_neg)
        positives = get_list_couples(d_pos)
        for pair in positives:
            rgx = re.search(r"(.+)_(.+)", pair)
            first_gene, second_gene = rgx.group(1), rgx.group(2)
            gene_to_search = second_gene if first_gene == controlled_gene else first_gene
            negative_pair =  get_couple_id(control_gene, gene_to_search)
            if negative_pair in negatives:
                diz[idx] = {'positive': pair, 'negative':negative_pair}
            else:
                diz[idx] = {'positive': pair, 'negative':np.nan}
            idx+=1
        
    df_pairs = pd.DataFrame.from_dict(diz, 'index')
    return df_pairs

def swap_genes_if_needed(df):
    original_dim = df.shape[0]
    where = df.need_to_swap
    df.loc[where, ['gene_id1', 'gene_id2']] = (df.loc[where, ['gene_id2', 'gene_id1']].values)
    df.loc[where, ['length_1', 'length_2']] = (df.loc[where, ['length_2', 'length_1']].values)
    df.loc[where, ['start_map1', 'start_map2']] = (df.loc[where, ['start_map2', 'start_map1']].values)
    df.loc[where, ['end_map1', 'end_map2']] = (df.loc[where, ['end_map2', 'end_map1']].values)
    df.loc[where, ['transcript_biotype_1', 'transcript_biotype_2']] = (df.loc[where, ['transcript_biotype_2', 'transcript_biotype_1']].values)
    df.loc[where, ['gene_type1', 'gene_type2']] = (df.loc[where, ['gene_type2', 'gene_type1']].values)
    df.loc[where, ['cdna_1', 'cdna_2']] = (df.loc[where, ['cdna_2', 'cdna_1']].values)
    df = df.drop('need_to_swap', axis = 1)
    df = df.drop_duplicates().reset_index(drop = True)
    n_duplicates = original_dim - df.shape[0] 
    print(f"{n_duplicates} interactions were duplicated (the genes were swopped, now they have a unique couples_id so I can see only now that they are duplicated)")
    return df

def create_features(df):
    df['area_of_the_matrix'] = df['length_1']*df['length_2']
    df['protein_coding_1'] = df['transcript_biotype_1'].apply(lambda x: True if x == 'protein_coding' else False)
    df['protein_coding_2'] = df['transcript_biotype_2'].apply(lambda x: True if x == 'protein_coding' else False)
    assert (df['end_map1'] - df['start_map1']).min() > 0 
    assert (df['end_map2'] - df['start_map2']).min() > 0 
    df['area_of_the_interaction'] = (df['end_map1'] - df['start_map1'])*(df['end_map2'] - df['start_map2'])
    df['interacting'] = True
    assert set(pd.get_dummies(df.cell_line).columns) == set(['HEK293T', 'HeLa', 'HEK293', 'mES', 'Mouse_brain'])
    df = pd.concat([df, pd.get_dummies(df.cell_line)], axis = 1).drop('cell_line', axis = 1)
    return df

def create_boxes_xywh(row):
    """
    start_map1, end_map1, start_map2, end_map2 are in the interval [0, len(rna)-1] for the interactive pairs (indeed, the interactive regions must be sliced like: cdna1[start_map1:(end_map1-1)])
    Args:
        a row (pd.Series) of the dataset.
    Returns:
        boxes (list): A list of bboxes  with the form -> bbox = [x, y, w, h] 
    """
    x = row.start_map1
    y = row.start_map2
    w = row.end_map1 - row.start_map1
    h = row.end_map2 - row.start_map2
    assert row.area_of_the_interaction == h*w
    # if row.area_of_the_interaction != h*w:
    #     print(row.area_of_the_interaction, h*w, h, w)
    #     assert False
    
    return pd.Series([x, y, w, h])

#  - - - - - - - - - - - - - - Create fake coordinates for negatives - - - - - - - - - - - - - - - 

def where_interacts(x):
    if x.protein_coding == False:
        return 'all'
    elif x.c1 <= x.UTR5:
        return 'UTR5'
    elif (x.c1 > x.UTR5) & (x.c1 <= x.CDS):
        return 'CDS'
    else:
        return 'UTR3'

def create_fake_coord_neg(x, df_coord, df_pairs_full, df_int, max_num_tries = 100):
    g1 = x.gene1
    g2 = x.gene2
    for _ in range(max_num_tries):
        s1 = df_coord[df_coord.gene == g1].sample(1).iloc[0]
        s2 = df_coord[df_coord.gene == g2].sample(1).iloc[0]
        if ((s1.where_c1 == 'CDS')&(s2.where_c1 == 'CDS')) == False:
            break
        #it can be CDS-CDS only if (s1.where_c1 == 'CDS')&(s2.where_c1 == 'CDS') after max_num_tries iterations
    
    pos = df_pairs_full[df_pairs_full.negative == x.couples].sample(1).iloc[0] #dovrebbe essere 1 ma non sono sicuro (e possibile piu di una? dovrei ragionarci su con l assert di prima), nel dubbio campiono.
    p1, p2 = pos.positive.split('_')
    
    interaction_coords = df_int[df_int.couples == pos.positive].sample(1).iloc[0] #puo essere piu di una, se ho piu di una regione di interzione
    
    if g1 == p1:
        x1, w = interaction_coords.x1, interaction_coords.w
        y1, h = s2.c1, s2.l
        
    elif g1 == p2:
        x1, w = interaction_coords.y1, interaction_coords.h
        y1, h = s2.c1, s2.l
        
    elif g2 == p1:
        x1, w = s1.c1, s1.l
        y1, h = interaction_coords.x1, interaction_coords.w
        
    elif g2 == p2:
        x1, w = s1.c1, s1.l
        y1, h = interaction_coords.y1, interaction_coords.h
        
    else:
        raise NotImplementedError
    
    return x1, y1, w, h

#  - - - - - - - - - - - - - - Clean bounding boxes of df interactions - - - - - - - - - - - - - - 

def clean_bounding_boxes(list_bboxes):
        list_of_indexes = []
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
        return new_list_of_boxes

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