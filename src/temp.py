import pandas as pd
import os
import time
import numpy as np
import sys
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset.preprocessing as utils
from config import *

def order_genes(x):
    x1, x2 = x.split('_')
    return utils.get_couple_id(x1, x2)

def main():

    df_hub = utils.read_dataframe(os.path.join(original_files_dir, 'hub.table.paris.txt'), columns_to_drop = ['Unnamed: 0','gene_name']).rename({'cell_line': 'cell_line_set', 'degree':'n_interactors', 'gene_type': 'gene_type_set', 'species': 'species_set'}, axis = 1)
    tx = utils.read_dataframe(os.path.join(original_files_dir,'tx_regions.ens99.txt'), columns_to_drop = ['Unnamed: 0','ensembl_transcript_id']).rename({'ensembl_gene_id': 'gene_id'}, axis = 1)
    df_genes = df_hub.merge(tx, on = 'gene_id')
    assert df_genes.shape[0] == df_hub.shape[0]


    
    filename = os.path.join(original_files_dir, 'Tx.RI_ALL.specie.no_abundant.filter_rep.no_mirna.no_mito.no_cds_cds.bedpe')

    rows = []
    with open(filename)as f:
        for line in f:
            L = line.strip().split()
            rows.append(L)

    all_interactions = pd.DataFrame(rows, columns = ['tr1', 'x1', 'x2', 'tr2', 'y1', 'y2', 'name', 'n_reads', 'strand1', 'strand2', 'other_id', 'experiment'] )

    #they are already mapped so the strand should be always '+'
    assert (all_interactions.strand1 == all_interactions.strand2).all()
    assert set(all_interactions.strand1) == {'+'}

    #all_interactions = all_interactions.filter(['tr1', 'x1', 'x2', 'tr2', 'y1', 'y2', 'experiment', 'name'], axis = 1)

    #all_interactions[['x1', 'x2', 'y1', 'y2']] = all_interactions[['x1', 'x2', 'y1', 'y2']].apply(pd.to_numeric)

    all_interactions = all_interactions.filter(['experiment', 'name'], axis = 1)

    paris_experiments = ['hs_PARIS1__Hela_highRNase', 'mm_PARIS2__GSM4503873_Mouse_brain_mRNA', 'hs_PARIS1__Hela_lowRNase', 'hs_PARIS1__HEK293T', 'hs_PARIS2__GSM4503872_HEK293_mRNA', 'mm_PARIS1__mES', 'hs_PARIS2__HEK293_AMT', 'hs_PARIS2__HEK293_Amoto']
    mario_experiments = ['mm_MARIO__mES']
    ricseq_experiments = ['hs_RIC-seq__HeLa_merge.InterGene.arms']

    all_interactions = all_interactions[all_interactions['experiment'].isin(paris_experiments)].reset_index(drop = True)
    
    int_or = utils.read_dataframe(os.path.join(original_files_dir, 'rise_paris_tr.new.mapped_interactions.tx_regions.txt'), columns_to_drop = ['Unnamed: 0',  'Unnamed: 0.1', 'gene_name1', 'gene_name2', 'type_interaction', 'score', 'tx_id_1', 'tx_id_2', 'tx_id_1_localization', 'tx_id_2_localization', 'unique_id'])
    int_or = int_or.drop_duplicates().reset_index(drop = True)
    all_rise_id = set(int_or.rise_id)
    int_or = all_interactions.merge(int_or, left_on = 'name', right_on = 'rise_id').drop('name', axis = 1)
    assert all_rise_id == set(int_or.rise_id)
    
    #accorpiamo alcuni esperimenti per generare i negativi 
    #[hs_PARIS2__HEK293_AMT, hs_PARIS2__HEK293_Amoto]
    #[hs_PARIS1__Hela_highRNase, hs_PARIS1__Hela_lowRNase]

    all_interactions.loc[all_interactions["experiment"] == "hs_PARIS2__HEK293_AMT", "experiment"] = "hs_PARIS2__HEK293"
    all_interactions.loc[all_interactions["experiment"] == "hs_PARIS2__HEK293_Amoto", "experiment"] = "hs_PARIS2__HEK293"

    int_or.loc[int_or["experiment"] == "hs_PARIS2__HEK293_AMT", "experiment"] = "hs_PARIS2__HEK293"
    int_or.loc[int_or["experiment"] == "hs_PARIS2__HEK293_Amoto", "experiment"] = "hs_PARIS2__HEK293"


    all_interactions.loc[all_interactions["experiment"] == "hs_PARIS1__Hela_highRNase", "experiment"] = "hs_PARIS1__Hela"
    all_interactions.loc[all_interactions["experiment"] == "hs_PARIS1__Hela_lowRNase", "experiment"] = "hs_PARIS1__Hela"

    int_or.loc[int_or["experiment"] == "hs_PARIS1__Hela_highRNase", "experiment"] = "hs_PARIS1__Hela"
    int_or.loc[int_or["experiment"] == "hs_PARIS1__Hela_lowRNase", "experiment"] = "hs_PARIS1__Hela"

    paris_experiments = ['hs_PARIS1__Hela', 'mm_PARIS2__GSM4503873_Mouse_brain_mRNA', 'hs_PARIS1__HEK293T', 'hs_PARIS2__GSM4503872_HEK293_mRNA', 'mm_PARIS1__mES', 'hs_PARIS2__HEK293']
    
    how_many_negatives_per_positive = 2 

    df_pairs_full = []
    for exp in paris_experiments:
        df_exp = int_or[int_or.experiment == exp][['gene_id1', 'gene_id2']].drop_duplicates().reset_index(drop = True)
        df_exp['positive'] = df_exp['gene_id1'] + '_' + df_exp['gene_id2']
        for i in range(how_many_negatives_per_positive):
            df_exp['gene_id2'] = df_exp.sample(frac=1, replace=False, random_state=42).reset_index(drop = True)['gene_id2']
            df_exp[f'negative{i}'] = df_exp['gene_id1'] + '_' + df_exp['gene_id2']
        df_exp['experiment'] = exp
        df_pairs_full.append(df_exp[['positive', 'negative0', 'negative1', 'experiment']])
    df_pairs_full = pd.concat(df_pairs_full, axis = 0)
    assert set(df_pairs_full.positive) == set(int_or['gene_id1'] + '_' + int_or['gene_id2'])

    df_pairs_full = pd.concat([
        df_pairs_full[['positive', 'negative0', 'experiment']].rename({'negative0':'negative'}, axis = 1),
        df_pairs_full[['positive', 'negative1', 'experiment']].rename({'negative1':'negative'}, axis = 1)
    ], axis = 0)

    df_pairs_full['negative'] = df_pairs_full['negative'].apply(order_genes)
    df_pairs_full['positive'] = df_pairs_full['positive'].apply(order_genes)

    assert set(df_pairs_full.positive) == set((int_or['gene_id1'] + '_' + int_or['gene_id2']).apply(order_genes))
    
    #Drop all the negative interactions that are seen positive in PARIS or in other experiments
    
    prohibited_couples = pd.read_csv(os.path.join(original_files_dir, 'prohibited_couples.txt'), sep = '\t')
    prohibited_couples['messy_id'] = prohibited_couples['gene_id1'] + '_' + prohibited_couples['gene_id2']
    prohibited_couples['id'] = prohibited_couples['messy_id'].apply(order_genes)
    print(len(set(df_pairs_full.negative)))

    df_pairs_full = df_pairs_full[~df_pairs_full.negative.isin(prohibited_couples['id'])].reset_index(drop = True)
    print(len(set(df_pairs_full.negative)))
    
    to_drop = set(df_pairs_full['positive']).intersection(df_pairs_full['negative'])

    print('how many to drop', len(to_drop))

    df_pairs_full = df_pairs_full[~df_pairs_full['negative'].isin(to_drop)]
    
    print(f'We have {len(set(df_pairs_full.positive))} pairs interacting (they can have multiple interactions) \n')
    print(f'We have {len(set(df_pairs_full.negative))} pairs not interacting \n')
    
    
    # FIXING df_pairs_full
    # it can happen that few positive samples are not included in df_pairs_full because I have excluded some rows

    int_or[['couples', 'need_to_swap']] = int_or[['gene_id1', 'gene_id2']].apply(utils.create_pairs, axis = 1)

    not_included = set(int_or.couples) - set(df_pairs_full.positive).union(set(df_pairs_full.negative))

    int_or_not_incl = int_or[int_or.couples.isin(not_included)]
    int_or_not_incl = int_or[int_or.couples.isin(not_included)][['couples', 'experiment']]
    int_or_not_incl = int_or_not_incl.drop_duplicates(subset = 'couples')
    int_or_not_incl = int_or_not_incl.reset_index(drop = True)
    c = int_or_not_incl.couples.str.extractall('(.*)_(.*)').reset_index()
    int_or_not_incl['gene1'], int_or_not_incl['gene2'] = c[0], c[1]



    df_pairs_full2 = []
    for _, row in int_or_not_incl.iterrows():
        subset_df_pairs_full = df_pairs_full[df_pairs_full.experiment == row.experiment].reset_index()

        p = subset_df_pairs_full.positive.str.extractall('(.*)_(.*)').reset_index()
        n = subset_df_pairs_full.negative.str.extractall('(.*)_(.*)').reset_index()
        subset_df_pairs_full['p0'], subset_df_pairs_full['p1'] = p[0], p[1]
        subset_df_pairs_full['n0'], subset_df_pairs_full['n1'] = n[0], n[1]

        subset_df_pairs_full = subset_df_pairs_full[
                (subset_df_pairs_full['n0'] == row.gene1)|(subset_df_pairs_full['n0'] == row.gene2)|
                (subset_df_pairs_full['n1'] == row.gene1)|(subset_df_pairs_full['n1'] == row.gene2)
        ] 
        neg_sampled = subset_df_pairs_full.sample(1).negative.iloc[0]
        df_pairs_full2.append({'positive': row.couples, 'negative':neg_sampled, 'experiment': exp})

    df_pairs_full = pd.concat([df_pairs_full, pd.DataFrame(df_pairs_full2)], axis = 0).reset_index(drop = True)

    assert set(int_or.couples) - set(df_pairs_full.positive).union(set(df_pairs_full.negative)) == set()
    
    
    
    # EXPORT
    df_pairs_full.to_csv(os.path.join(processed_files_dir, 'df_pairs_full_RANDOM.csv'), index = False)
    
    df_genes = pd.read_csv(os.path.join(processed_files_dir, 'df_genes.csv'))
    
    df_int = pd.read_csv(os.path.join(processed_files_dir, 'full_paris_info_interactions.csv'))

    df_int = df_int[['couples', 'gene1', 'gene2', 
                     'interacting', 'length_1', 'length_2',
                     'protein_coding_1', 'protein_coding_2',
                     'x1', 'y1', 'w', 'h']]
    
    df_neg = df_pairs_full[['negative']].drop_duplicates().reset_index(drop = True)
    df_neg[['gene1', 'gene2']] = df_neg['negative'].str.split('_', expand = True)
    df_neg = df_neg.rename({'negative':'couples'}, axis = 1)

    df_neg['interacting'] = False
    df_neg = df_neg.merge(df_genes[['gene_id', 'length', 'protein_coding']], left_on = 'gene1', right_on = 'gene_id').drop('gene_id', axis = 1).rename({'length': 'length_1','protein_coding':'protein_coding_1'} , axis = 1)
    df_neg = df_neg.merge(df_genes[['gene_id', 'length', 'protein_coding']], left_on = 'gene2', right_on = 'gene_id').drop('gene_id', axis = 1).rename({'length': 'length_2','protein_coding':'protein_coding_2'} , axis = 1)


    assert (set(df_pairs_full.negative) - set(df_neg.couples) == {np.nan})|(set(df_pairs_full.negative) - set(df_neg.couples) == set()) # I have some NaN in the df_pairs_full

    df_int1 = df_int[['gene1', 'x1', 'w']].rename({'gene1':'gene', 'x1':'c1',  'w': 'l'}, axis = 1)
    df_int2 = df_int[['gene2', 'y1', 'h']].rename({'gene2':'gene', 'y1':'c1',  'h': 'l'}, axis = 1)
    df_coord = pd.concat([df_int1, df_int2], ignore_index = True)#.drop_duplicates().reset_index(drop = True)
    #df_coord may have duplicates. but this is something I want. If a gene appears more than once, I want it to be sampled according to its distribution.
    
    assert set(df_neg.gene1).union(set(df_neg.gene2)) - set(df_coord.gene) == set()
    
    df_coord = df_coord.merge(
    df_genes.filter(['gene_id', 'UTR5', 'CDS', 'UTR3', 'protein_coding'], axis = 1).rename({'gene_id':'gene'}, axis = 1)
)
    df_coord['where_c1'] = df_coord.apply(utils.where_interacts, axis = 1)
    
    #127 min
    start_time = time.time()
    new_cols = df_neg[['couples', 'gene1', 'gene2']].apply(utils.create_fake_coord_neg, axis = 1, args = (df_coord,df_pairs_full,df_int,))
    print(f"Total time: {(time.time()-start_time)/60} minutes")
    
    new_cols = new_cols.apply(pd.Series).rename({0:'x1', 1:'y1', 2:'w', 3:'h'}, axis = 1)
    df_neg = pd.concat([df_neg, new_cols], axis = 1)
    df = pd.concat([df_int, df_neg], ignore_index = True, axis = 0)
    
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
    print(f"Total time: {(time.time()-start_time)/60} minutes")
    
    df.to_csv(os.path.join(processed_files_dir, 'final_df_RANDOM.csv'), index = False)
    
    ### ADD EXPERIMENT COLUMN

    df_int, df_neg = df[df.interacting],  df[df.interacting == False]

    int_or = int_or.filter(['experiment', 'start_map1', 'end_map1', 'start_map2', 'end_map2', 'rise_id', 'gene_id1', 'gene_id2', 'method'], axis = 1)
    int_or = int_or.rename({'start_map1':'x1', 'end_map1':'x2', 'start_map2':'y1', 'end_map2':'y2'}, axis = 1)

    int_or[['couples', 'need_to_swap']] = int_or[['gene_id1', 'gene_id2']].apply(utils.create_pairs, axis = 1)

    def swap_genes_if_needed(df):
        original_dim = df.shape[0]
        where = df.need_to_swap
        df.loc[where, ['gene_id1', 'gene_id2']] = (df.loc[where, ['gene_id2', 'gene_id1']].values)
        df.loc[where, ['x1', 'y1']] = (df.loc[where, ['y1', 'x1']].values)
        df.loc[where, ['x2', 'y2']] = (df.loc[where, ['y2', 'x2']].values)
        df = df.drop('need_to_swap', axis = 1)
        df = df.drop_duplicates().reset_index(drop = True)
        n_duplicates = original_dim - df.shape[0] 
        return df

    int_or = swap_genes_if_needed(int_or)
    int_or['w'] = int_or['x2'] -  int_or['x1']
    int_or['h'] = int_or['y2'] -  int_or['y1']
    int_or = int_or.drop(['x2', 'y2', 'gene_id1', 'gene_id2'], axis = 1)

    merged = df_int.merge(int_or, on = ['couples', 'x1', 'w', 'y1', 'h']).reset_index(drop = True)

    # If I have multiple interactions of different experiments mapping to the same region, I keep only one of them
    merged = merged.drop_duplicates(subset = ['couples', 'x1', 'w', 'y1', 'h']).reset_index(drop = True)

    merged2 = df_int.merge(int_or, on = ['couples', 'x1', 'w', 'y1', 'h'], how = 'left').reset_index(drop = True)
    collapsed_int = merged2[merged2.experiment.isna()].reset_index(drop = True)
    collapsed_int = collapsed_int.fillna('unknwn_collapsed_interactions')

    assert merged.shape[0] + collapsed_int.shape[0] == df_int.shape[0]

    assert set(df_int.couples) == set(collapsed_int.couples).union(merged.couples)

    # add to positive df
    df_int = pd.concat([merged, collapsed_int], axis = 0)

    # add to negative df

    df_neg['rise_id'] = 'is_negative'
    df_neg = df_neg.merge(df_pairs_full[['negative', 'experiment']], left_on = 'couples', right_on = 'negative').drop('negative', axis = 1)
    df_neg['method'] = df_neg['experiment'].str.extractall('(.*)_(.*)__(.*)').reset_index()[1]
    df_neg = df_neg.filter(list(df_int.columns), axis = 1)

    # If I have multiple interactions of different experiments mapping to the same region, I keep only one of them
    df_neg = df_neg.drop_duplicates(subset = ['couples', 'x1', 'w', 'y1', 'h']).reset_index(drop = True)

    df = pd.concat([df_int, df_neg], ignore_index = True, axis = 0)
    
    mouse_genes = list(df_genes[df_genes.species_set == "{'mm'}"].gene_id)

    df['specie'] = 'human'

    df.loc[ (df.gene1.isin(mouse_genes))&(df.gene2.isin(mouse_genes)), 'specie'] = 'mouse'

    assert set(df[(df.gene1.isin(mouse_genes))&(df.gene2.isin(mouse_genes))].specie) == set({'mouse'})
    
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(os.path.join(processed_files_dir, 'final_df_RANDOM.csv'), index = False)

if __name__ == '__main__':
    #run me with: -> 
    #nohup python temp.py &> temp.out &

    main()