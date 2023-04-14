import pandas as pd
import os
import time
import numpy as np
import sys
from tqdm.notebook import tqdm
sys.path.insert(0, '..')
import dataset.preprocessing as utils

def main():
    start_time_script = time.time()

    df_hub = utils.read_dataframe(os.path.join(original_files_dir, 'hub.table.paris.txt'), columns_to_drop = ['Unnamed: 0','gene_name']).rename({'cell_line': 'cell_line_set',
                                                                                                                                                    'degree':'n_interactors',
                                                                                                                                                    'gene_type': 'gene_type_set',
                                                                                                                                                        'species': 'species_set'}, axis = 1)
    tx = utils.read_dataframe(os.path.join(original_files_dir,'tx_regions.ens99.txt'), columns_to_drop = ['Unnamed: 0','ensembl_transcript_id']).rename({'ensembl_gene_id': 'gene_id'}, axis = 1)
    cc = utils.read_dataframe(os.path.join(original_files_dir,'controls_controlled.hub.txt'), columns_to_drop = ['Unnamed: 0'])
    int_or = utils.read_dataframe(os.path.join(original_files_dir, 'rise_paris_tr.new.mapped_interactions.tx_regions.txt'), columns_to_drop = ['Unnamed: 0',  'Unnamed: 0.1', 'gene_name1', 'gene_name2', 'score', 'tx_id_1', 'tx_id_2', 'rise_id', 'type_interaction', 'tx_id_1_localization', 'tx_id_2_localization'])
    int_or = int_or.drop_duplicates().reset_index(drop = True)
    df_genes = df_hub.merge(tx, on = 'gene_id')
    assert df_genes.shape[0] == df_hub.shape[0]
    df_pairs_full = utils.obtain_df_pos_controls(cc)
    print(f'We have {len(set(df_pairs_full.positive))} pairs interacting (they can have multiple interactions) \n')
    print(f'We have {len(set(df_pairs_full.negative))} pairs not interacting \n')
    df_pairs_full.to_csv(os.path.join(processed_files_dir, 'df_pairs_full.csv'), index = False)

    assert len(set(df_pairs_full.positive).intersection(set(int_or.gene_id1 + '_' + int_or.gene_id2))) > 0
    assert len(set(df_pairs_full.positive).intersection(set(int_or.gene_id2 + '_' + int_or.gene_id1))) > 0
    assert len(set(df_pairs_full.negative).intersection(set(int_or.gene_id1 + '_' + int_or.gene_id2))) == 0
    assert len(set(df_pairs_full.negative).intersection(set(int_or.gene_id2 + '_' + int_or.gene_id1))) == 0
    
    df_neg = df_pairs_full[['negative']].drop_duplicates().reset_index(drop = True)
    df_neg[['gene1', 'gene2']] = df_neg['negative'].str.split('_', expand = True)
    df_neg = df_neg.rename({'negative':'couples'}, axis = 1)

    df_pairs = df_pairs_full.groupby('positive').agg({'negative': list}).reset_index()
    int_or[['couples', 'need_to_swap']] = int_or[['gene_id1', 'gene_id2']].apply(utils.create_pairs, axis = 1)
    int_or = utils.swap_genes_if_needed(int_or)
    assert (int_or[['gene_id1', 'gene_id2']].apply(utils.create_pairs, axis = 1)[1] == False).all() #check if swapping works
    int_or = utils.create_features(int_or)
    assert int_or.groupby('gene_id1').std(numeric_only = True).protein_coding_1.max() == 0
    assert int_or.groupby('gene_id2').std(numeric_only = True).protein_coding_2.max() == 0
    assert int_or.groupby('gene_id1').std(numeric_only = True).length_1.max() == 0
    assert int_or.groupby('gene_id2').std(numeric_only = True).length_2.max() == 0
    idx = np.random.randint(int_or.shape[0])
    assert int_or.loc[idx].length_1 == len(int_or.loc[idx].cdna_1)
    assert int_or.loc[idx].length_2 == len(int_or.loc[idx].cdna_2)
    gene_info1 = int_or[['gene_id1', 'length_1', 'cdna_1', 'protein_coding_1']]
    gene_info1.columns = ['gene_id', 'length', 'cdna', 'protein_coding']
    gene_info2 = int_or[['gene_id2', 'length_2', 'cdna_2', 'protein_coding_2']]
    gene_info2.columns = ['gene_id', 'length', 'cdna', 'protein_coding']
    gene_info = pd.concat([gene_info1, gene_info2], axis = 0, ignore_index = True).drop_duplicates()
    #assert set(gene_info.gene_id) == set(df_genes.gene_id)
    df_genes = df_genes.merge(gene_info)
    df_genes.to_csv(os.path.join(processed_files_dir, 'df_genes.csv'), index = False)
    #clean int_or
    int_or = int_or.drop(['cdna_1', 'cdna_2'], axis = 1)
    df_boxes = int_or.filter(['start_map1', 'end_map1', 'start_map2', 'end_map2','area_of_the_interaction'], axis = 1).apply(utils.create_boxes_xywh, axis = 1).rename({0: 'x1', 1: 'y1', 2:'w', 3:'h'}, axis = 1)
    int_or = pd.concat([int_or, df_boxes], axis = 1).drop(['start_map1', 'end_map1', 'start_map2', 'end_map2'], axis = 1)
    #approx 11 min
    diz_int = {}
    idx = 0
    for couple in tqdm(int_or.couples.unique()):
        subset = int_or[int_or.couples == couple]
        list_of_boxes = subset.filter(['x1', 'y1', 'w', 'h']).values.tolist()
        new_list_of_boxes = utils.clean_bounding_boxes(list_of_boxes)
        row = int_or[int_or.couples == couple].iloc[0]
        for box in new_list_of_boxes:
            d = dict(row)
            d['x1'] = box[0]
            d['y1'] = box[1] 
            d['w'] = box[2]
            d['h'] = box[3]
            diz_int[idx] = d
            idx+=1
    df_int = pd.DataFrame.from_dict(diz_int, 'index').rename({'gene_id1':'gene1', 'gene_id2':'gene2'}, axis = 1)
    assert len(int_or.couples.unique()) == len(df_int.couples.unique())
    print(f'#interazioni prima {int_or.shape[0]}, #interazioni dopo: {df_int.shape[0]}')
    df_int.to_csv(os.path.join(processed_files_dir, 'full_paris_info_interactions.csv'), index = False)
    df_int = df_int[['couples', 'gene1', 'gene2', 
                 'interacting', 'length_1', 'length_2',
                 'protein_coding_1', 'protein_coding_2',
                 'x1', 'y1', 'w', 'h']]
    df_neg['interacting'] = False
    df_neg = df_neg.merge(df_genes[['gene_id', 'length', 'protein_coding']], left_on = 'gene1', right_on = 'gene_id').drop('gene_id', axis = 1).rename({'length': 'length_1','protein_coding':'protein_coding_1'} , axis = 1)
    df_neg = df_neg.merge(df_genes[['gene_id', 'length', 'protein_coding']], left_on = 'gene2', right_on = 'gene_id').drop('gene_id', axis = 1).rename({'length': 'length_2','protein_coding':'protein_coding_2'} , axis = 1)
    assert set(df_pairs_full.negative) - set(df_neg.couples) == {np.nan} # I have some NaN in the df_pairs_full
    df_int1 = df_int[['gene1', 'x1', 'w']].rename({'gene1':'gene', 'x1':'c1',  'w': 'l'}, axis = 1)
    df_int2 = df_int[['gene2', 'y1', 'h']].rename({'gene2':'gene', 'y1':'c1',  'h': 'l'}, axis = 1)
    df_coord = pd.concat([df_int1, df_int2], ignore_index = True)#.drop_duplicates().reset_index(drop = True)
    #df_coord may have duplicates. but this is something I want. If a gene appears more than once, I want it to be sampled according to its distribution.
    assert set(df_neg.gene1).union(set(df_neg.gene2)) - set(df_coord.gene) == set()
    #65 min
    start_time = time.time()
    new_cols = df_neg[['couples', 'gene1', 'gene2']].apply(utils.create_fake_coord_neg, axis = 1, args = (df_coord,df_pairs_full,df_int,))
    print(f"Total time create_fake_coord_neg: {(time.time()-start_time)/60} minutes")
    new_cols = new_cols.apply(pd.Series).rename({0:'x1', 1:'y1', 2:'w', 3:'h'}, axis = 1)
    df_neg = pd.concat([df_neg, new_cols], axis = 1)
    df = pd.concat([df_int, df_neg], ignore_index = True, axis = 0)
    df.to_csv(os.path.join(processed_files_dir, 'final_df.csv'), index = False)
    print(f"Final dataframe exported, now I do checks")
    
    #check if it worked
    assert (df_neg.x1 <= df_neg.length_1).all()
    assert ((df_neg.x1 + df_neg.w) <= df_neg.length_1).all()
    assert (df_neg.y1 <= df_neg.length_2).all()
    assert ((df_neg.y1 + df_neg.h) <= df_neg.length_2).all()
    
    # 37 min
    #check if it worked
    start_time = time.time()
    for _, row in df_neg.iterrows():
        g1 = row.gene1
        g2 = row.gene2
        assert [row.x1, row.w] in df_coord[df_coord.gene == g1][['c1', 'l']].values
        assert [row.y1, row.h] in df_coord[df_coord.gene == g2][['c1', 'l']].values
        if np.random.rand() < 0.0003: #progress
            print(f"{np.round(_/df_neg.shape[0] * 100, 2)}% in {(time.time()-start_time)/60} minutes")
    print(f"Total time to check: {(time.time()-start_time)/60} minutes")
    

    print(f"Total time: {(time.time()-start_time_script)/60} minutes")
    
    
if __name__ == '__main__':
    #run me with: -> 
    #nohup python preprocess_adri_data.py &> preprocess_adri_data.out &

    ROOT_DIR = os.path.dirname(os.path.abspath('.'))
    original_files_dir = os.path.join(ROOT_DIR, 'dataset', 'original_files')
    processed_files_dir = os.path.join(ROOT_DIR, 'dataset', 'processed_files')
    rna_rna_pairs_data_dir = os.path.join(ROOT_DIR, 'dataset', 'rna_rna_pairs')
    
    main()
