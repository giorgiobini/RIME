import pandas as pd
import ast
import numpy as np
from sklearn.metrics import mean_absolute_error
import os.path
import time
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from matplotlib_venn import venn3

def read_dataframe(file_path, columns_to_drop = []):
    cols = list(pd.read_csv(file_path, sep ='\t', nrows =1))
    # Use list comprehension to remove the unwanted column in **usecol**
    df = pd.read_csv(file_path, sep ='\t', usecols =[i for i in cols if i not in columns_to_drop])
    return df

def merge_features(df_int, df_con):
    """
    I only need this merge because the controls dataframe does not have the same features of the interactors one
    """
    df_int1 = df_int[['ensembl_transcript_id_1', 'transcript_biotype_1', 'species']].drop_duplicates().rename({'ensembl_transcript_id_1':'ensembl_transcript_id',  'transcript_biotype_1':'transcript_biotype'}, axis =1)
    df_int2 = df_int[['ensembl_transcript_id_2', 'transcript_biotype_2', 'species']].drop_duplicates().rename({'ensembl_transcript_id_2':'ensembl_transcript_id', 'transcript_biotype_2':'transcript_biotype'}, axis =1)
    df_transcripts = pd.concat([df_int1, df_int2], axis = 0).drop_duplicates().reset_index(drop = True)
    df_con_merged = df_con.merge(df_transcripts, left_on = ['ensembl_transcript_id_1'], right_on = ['ensembl_transcript_id']).rename({'transcript_biotype':'transcript_biotype_1'}, axis = 1)                                                                                                             
    df_con_merged = df_con_merged.merge(df_transcripts, left_on = ['ensembl_transcript_id_2', 'species'], right_on = ['ensembl_transcript_id', 'species']).rename({'transcript_biotype':'transcript_biotype_2'}, axis = 1)                                                                                                                      
    df_con_merged = df_con_merged.drop(['ensembl_transcript_id_x', 'ensembl_transcript_id_y'], axis = 1)
    assert df_con_merged.shape[0] == df_con.shape[0]
    return df_con_merged

def create_features(df, df_hub, df_interactors):
    df['area_of_the_matrix'] = df['length_1']*df['length_2']
    df['protein_coding_1'] = df['transcript_biotype_1'].apply(lambda x: True if x == 'protein_coding' else False)
    df['protein_coding_2'] = df['transcript_biotype_2'].apply(lambda x: True if x == 'protein_coding' else False)
    if df_interactors:
        assert df['width_map1'].min() > 0 
        assert df['width_map2'].min() > 0 
        df['area_of_the_interaction'] = df['width_map1']*df['width_map2']
        df['there_is_interaction'] = True
        df = pd.concat([df, pd.get_dummies(df.cell_line)], axis = 1).drop('cell_line', axis = 1)
    else:
        df['there_is_interaction'] = False
        df['area_of_the_interaction'] = 0
        df['start_map1'] = 0
        df['end_map1'] = 0
        df['start_map2'] = 0
        df['end_map2'] = 0
        df['tx_id_1_localization'] = np.nan
        df['tx_id_2_localization'] = np.nan
        df = get_cell_line_of_non_interactors(df, df_hub)
    df[['couples', 'need_to_swap']] = df[['gene_id1', 'gene_id2']].apply(create_pairs, axis = 1)
    df = df.filter(['couples', 'gene_id1', 'gene_id2', 
    'ensembl_transcript_id_1', 'ensembl_transcript_id_2', 
    'length_1', 'length_2', 'transcript_biotype_1', 'transcript_biotype_2',
    'protein_coding_1', 'protein_coding_2', 'there_is_interaction', 
    'area_of_the_matrix', 'area_of_the_interaction', 'species',
    'HEK293T', 'Hela(highRNase)', 'Hela(lowRNase)','mES', 
    'start_map1', 'end_map1', 'start_map2', 'end_map2',
    'tx_id_1_localization', 'tx_id_2_localization', 'need_to_swap'], axis = 1)
    df = swap_genes_if_needed(df)
    return df

def swap_genes_if_needed(df):
    where = df.need_to_swap
    df.loc[where, ['gene_id1', 'gene_id2']] = (df.loc[where, ['gene_id2', 'gene_id1']].values)
    df.loc[where, ['length_1', 'length_2']] = (df.loc[where, ['length_2', 'length_1']].values)
    df.loc[where, ['start_map1', 'start_map2']] = (df.loc[where, ['start_map2', 'start_map1']].values)
    df.loc[where, ['end_map1', 'end_map2']] = (df.loc[where, ['end_map2', 'end_map1']].values)
    df.loc[where, ['protein_coding_1', 'protein_coding_2']] = (df.loc[where, ['protein_coding_2', 'protein_coding_1']].values)
    df.loc[where, ['ensembl_transcript_id_1', 'ensembl_transcript_id_2']] = (df.loc[where, ['ensembl_transcript_id_2', 'ensembl_transcript_id_1']].values)
    df.loc[where, ['transcript_biotype_1', 'transcript_biotype_2']] = (df.loc[where, ['transcript_biotype_2', 'transcript_biotype_1']].values)
    df.loc[where, ['tx_id_1_localization', 'tx_id_2_localization']] = (df.loc[where, ['tx_id_2_localization', 'tx_id_1_localization']].values)
    return df.drop('need_to_swap', axis = 1)
    

def get_cell_line_of_non_interactors(df, df_hub):
    #I will take the intersection between the cell line sets of each gene of the pair. I need the df_hub for this
    new_cols = df.apply(get_intersection_set, df_hub = df_hub, axis = 1)
    new_cols = new_cols.apply(pd.Series)
    new_cols.columns = ['HEK293T', 'Hela(highRNase)', 'Hela(lowRNase)','mES']
    df = pd.concat([df, new_cols], axis = 1)
    #df['HEK293T'], df['Hela(highRNase)'], df['Hela(lowRNase)'], df['mES'] = df.apply(get_intersection_set, df_hub = df_hub, axis = 1)
    return df

def get_intersection_set(x, df_hub):
    set_gene1 = df_hub[df_hub.gene_id == x.gene_id1].cell_line_set.iloc[0]
    set_gene1 = list(ast.literal_eval(set_gene1))
    set_gene2 = df_hub[df_hub.gene_id == x.gene_id2].cell_line_set.iloc[0]
    set_gene2 = list(ast.literal_eval(set_gene2))
    intersection_set = list(set(set_gene1) & set(set_gene2))
    Hek = 1 if 'HEK293T' in intersection_set else 0
    Hela_highRNase = 1 if 'Hela(highRNase)' in intersection_set else 0
    Hela_lowRNase = 1 if 'Hela(lowRNase)' in intersection_set else 0
    mES = 1 if 'mES' in intersection_set else 0
    return Hek, Hela_highRNase, Hela_lowRNase, mES

def create_pairs(x):
    """
    Pairs will be created in such a way that gene1_gene2 is equal to gene2_gene1. How? I will simply order the strings before create the couple string.
    """
    first_gene = x.gene_id1
    second_gene = x.gene_id2
    l = sorted([first_gene, second_gene])
    need_to_swap = False if l[0] == first_gene else True
    return pd.Series(['_'.join(l), need_to_swap])

def create_df_hub(df_int, df_con):
    couples_duplicated, counts = np.unique(df_int[df_int.couples.duplicated()].couples, return_counts = True)
    counts += 1
    couples_duplicated_dict = dict(zip(couples_duplicated, counts))
    df_con1 = df_con[['gene_id1']].rename({'gene_id1':'gene_id'}, axis =1)
    df_con2 = df_con[['gene_id2']].rename({'gene_id2':'gene_id'}, axis =1)
    df_con_full = pd.concat([df_con1, df_con2], axis = 0).reset_index(drop = True)
    df_con_full['n_not_interactors'] = 1
    df_con_full = df_con_full.groupby('gene_id').sum().reset_index()
    
    df_int1 = df_int[['gene_id1', 'couples']].rename({'gene_id1':'gene_id'}, axis =1)
    df_int2 = df_int[['gene_id2', 'couples']].rename({'gene_id2':'gene_id'}, axis =1)
    df_int_full = pd.concat([df_int1, df_int2], axis = 0).reset_index(drop = True)
    df_int_full['n_interactors'] = df_int_full.couples.apply(lambda x: 1/couples_duplicated_dict[x] if x in couples_duplicated_dict.keys() else 1)
    df_int_full = df_int_full.drop('couples', axis = 1)
    df_int_full = df_int_full.groupby('gene_id').sum().reset_index()

    df_full = df_int_full.merge(df_con_full, 'outer')
    df_full = df_full.replace(np.nan, 0)
    df_full.n_not_interactors = df_full.n_not_interactors.astype(int)
    df_full.n_interactors = df_full.n_interactors.astype(int)

    return df_full[['gene_id', 'n_interactors', 'n_not_interactors']]

def MSE(real, estimate):
    return np.square(real - estimate).mean()

def calc_target_loss1(df_int, df_con):
    assert len(df_int.species.value_counts()) == len(df_con.species.value_counts()) == 1
    specie = df_int.species.value_counts().index[0]
    assert specie in ['human', 'mouse']
    pr_cod_pr_cod_int = df_int[df_int.n_protein_codings_in_the_pair == 2].shape[0]/df_int.shape[0] #percentage of protein_coding-protein_coding in the interaction pair 
    pr_cod_no_pr_cod_int = df_int[df_int.n_protein_codings_in_the_pair == 1].shape[0]/df_int.shape[0] #percentage of protein_coding-non_protein_coding in the interaction pair
    no_pr_cod_no_pr_cod_int = df_int[df_int.n_protein_codings_in_the_pair == 0].shape[0]/df_int.shape[0] #percentage of non_protein_coding-non_protein_coding in the interaction pair is
    pr_cod_pr_cod_con = df_con[df_con.n_protein_codings_in_the_pair == 2].shape[0]/df_con.shape[0] #percentage of protein_coding-protein_coding in the non interaction pair
    pr_cod_no_pr_cod_con = df_con[df_con.n_protein_codings_in_the_pair == 1].shape[0]/df_con.shape[0] #percentage of protein_coding-non_protein_coding in the non interaction pair
    no_pr_cod_no_pr_cod_con = df_con[df_con.n_protein_codings_in_the_pair == 0].shape[0]/df_con.shape[0] #percentage of non_protein_coding-non_protein_coding in the non interaction pair
    return np.asarray([pr_cod_pr_cod_int, pr_cod_no_pr_cod_int, no_pr_cod_no_pr_cod_int, pr_cod_pr_cod_con, pr_cod_no_pr_cod_con, no_pr_cod_no_pr_cod_con])

def calc_target_loss2(df_int, df_con):
    assert len(df_int.species.value_counts()) == len(df_con.species.value_counts()) == 1
    specie = df_int.species.value_counts().index[0]
    assert specie == 'human'
    Hek_int = df_int['HEK293T'].sum()/df_int.shape[0] #ppercentage of HEK293T in the interaction pair
    Hela_highRNase_int = df_int['Hela(highRNase)'].sum()/df_int.shape[0] #percentage of Hela(highRNase) in the interaction pair
    Hela_lowRNase_int = df_int['Hela(lowRNase)'].sum()/df_int.shape[0] #percentage of Hela(lowRNase) in the interaction pair
    Hek_con = df_con['HEK293T'].sum()/df_con.shape[0] #percentage of HEK293T in the non interaction pair
    Hela_highRNase_con = df_con['Hela(highRNase)'].sum()/df_con.shape[0] #percentage of Hela(highRNase) in the non interaction pair
    Hela_lowRNase_con = df_con['Hela(lowRNase)'].sum()/df_con.shape[0] #percentage of Hela(lowRNase) in the non interaction pair 
    return np.asarray([Hek_int, Hela_highRNase_int, Hela_lowRNase_int, Hek_con, Hela_highRNase_con, Hela_lowRNase_con])


def print_stats(df):
    """
    df can contain human or mouse, not both
    """
    
    df_int, df_con = split_df_int_con(df)
    
    print('The percentage of interaction pair is: {}'.format(df_int.shape[0]/df.shape[0]))
    print('The percentage of non interaction pair is: {}'.format(df_con.shape[0]/df.shape[0]))
    
    pr_cod_pr_cod_int, pr_cod_no_pr_cod_int, no_pr_cod_no_pr_cod_int, pr_cod_pr_cod_con, pr_cod_no_pr_cod_con, no_pr_cod_no_pr_cod_con = calc_target_loss1(df_int, df_con)
    
    #I want approximately the same percentage of protein_coding-protein_coding, protein_coding-non_protein_coding, non_protein_coding-non_protein_coding interactions in the three sets.
    print('The percentage of protein_coding-protein_coding in the interaction pair is: {}'.format(pr_cod_pr_cod_int))
    print('The percentage of protein_coding-non_protein_coding in the interaction pair is: {}'.format(pr_cod_no_pr_cod_int))
    print('The percentage of non_protein_coding-non_protein_coding in the interaction pair is: {}'.format(no_pr_cod_no_pr_cod_int))
    print('The percentage of protein_coding-protein_coding in the non interaction pair is: {}'.format(pr_cod_pr_cod_con))
    print('The percentage of protein_coding-non_protein_coding in the non interaction pair is: {}'.format(pr_cod_no_pr_cod_con))
    print('The percentage of non_protein_coding-non_protein_coding in the non interaction pair is: {}'.format(no_pr_cod_no_pr_cod_con))
    print('\n')
    
    
    assert len(df_int.species.value_counts()) == len(df_con.species.value_counts()) == 1
    specie = df_int.species.value_counts().index[0]
    
    #I want approximately the same percentage of the different cell lines in the three sets.
    if specie == 'human':
        Hek_int, Hela_highRNase_int, Hela_lowRNase_int, Hek_con, Hela_highRNase_con, Hela_lowRNase_con = calc_target_loss2(df_int, df_con)
        print('The percentage of HEK293T in the interaction pair is: {}'.format(Hek_int))
        print('The percentage of Hela(highRNase) in the interaction pair is: {}'.format(Hela_highRNase_int))
        print('The percentage of Hela(lowRNase) in the interaction pair is: {}'.format(Hela_lowRNase_int))
        print('The percentage of HEK293T in the non interaction pair is: {}'.format(Hek_con))
        print('The percentage of Hela(highRNase) in the non interaction pair is: {}'.format(Hela_highRNase_con))
        print('The percentage of Hela(lowRNase) in the non interaction pair is: {}'.format(Hela_lowRNase_con))
    else:
        assert df[['HEK293T','Hela(highRNase)','Hela(lowRNase)']].sum().sum() == 0
        print('The only cell line is mES')
    print('\n') 
        
    #I want comparable matrix areas in the three sets.
    print('The mean area_of_the_matrix in the interaction pair is: {}'.format(df_int.area_of_the_matrix.mean()))
    print('The median area_of_the_matrix in the interaction pair is: {}'.format(df_int.area_of_the_matrix.median()))
    print('The standard deviation of area_of_the_matrix in the interaction pair is: {}'.format(df_int.area_of_the_matrix.std()))
    print('The mean area_of_the_matrix in the non interaction pair is: {}'.format(df_con.area_of_the_matrix.mean()))
    print('The median area_of_the_matrix in the non interaction pair is: {}'.format(df_con.area_of_the_matrix.median()))
    print('The standard deviation of area_of_the_matrix in the non interaction pair is: {}'.format(df_con.area_of_the_matrix.std()))
    print('\n')

    #I want comparable interaction areas in the three sets.
    print('The mean area_of_the_interaction in the interaction pair is: {}'.format(df_int.area_of_the_interaction.mean()))
    print('The median area_of_the_interaction in the interaction pair is: {}'.format(df_int.area_of_the_interaction.median()))
    print('The standard deviation of area_of_the_interaction in the interaction pair is: {}'.format(df_int.area_of_the_interaction.std()))
    print('\n')


def calc_bin_edges_loss3_4_5(column, number_of_bins = 10):
    bin_edges = np.quantile(column, np.linspace(0, 1, number_of_bins))
    return bin_edges

def calc_target_loss3_4_5(column, bin_edges):
    abs_val, bin_edges = np.histogram(np.asarray(column), bins=bin_edges)
    target_loss = abs_val/sum(abs_val)
    return target_loss

def split_df_int_con(df):
    return df[df.there_is_interaction == True].reset_index(drop = True), df[df.there_is_interaction == False].reset_index(drop = True)

def sort_df_hub(df_hub):
    df_hub['max_interactors'] = df_hub[['n_interactors','n_not_interactors']].max(axis=1)
    return df_hub.sort_values('max_interactors', ascending = False).drop('max_interactors', axis = 1).reset_index(drop = True)

def calc_loss(targets, predictions, specie):
    #Loss = Loss_1 + Loss_2 + mean(Loss_3 + Loss_4) + Loss_5
    if specie == 'human':
        target_loss1, target_loss2, target_loss3, target_loss4, target_loss5 = targets
        predicted_loss1, predicted_loss2, predicted_loss3, predicted_loss4, predicted_loss5 = predictions
        loss1 = mean_absolute_error(target_loss1, predicted_loss1)
        loss2 = mean_absolute_error(target_loss2, predicted_loss2)
        loss3 = mean_absolute_error(target_loss3, predicted_loss3)
        loss4 = mean_absolute_error(target_loss4, predicted_loss4)
        loss5 = mean_absolute_error(target_loss5, predicted_loss5)
    else:
        target_loss1, target_loss3, target_loss4, target_loss5 = targets
        predicted_loss1, predicted_loss3, predicted_loss4, predicted_loss5 = predictions
        loss1 = mean_absolute_error(target_loss1, predicted_loss1)
        loss2 = 0
        loss3 = mean_absolute_error(target_loss3, predicted_loss3)
        loss4 = mean_absolute_error(target_loss4, predicted_loss4)
        loss5 = mean_absolute_error(target_loss5, predicted_loss5)
    loss = loss1 + loss2 + np.mean(loss3+loss4) + loss5
    return loss

def calc_predictions_loss(df_int, df_con, specie, bin_edges):
    bin_edges_loss3, bin_edges_loss4, bin_edges_loss5 = bin_edges
    predicted_loss1 = calc_target_loss1(df_int, df_con)
    predicted_loss3 = calc_target_loss3_4_5(df_int.area_of_the_matrix, bin_edges_loss3)  #predicted_loss3 sums to 1
    predicted_loss4 =  calc_target_loss3_4_5(df_con.area_of_the_matrix, bin_edges_loss4)  #predicted_loss4 sums to 1
    predicted_loss5 = calc_target_loss3_4_5(df_int.area_of_the_interaction, bin_edges_loss5) #predicted_loss5 sums to 1
    if specie == 'human':
        predicted_loss2 = calc_target_loss2(df_int, df_con)
        predictions = predicted_loss1, predicted_loss2, predicted_loss3, predicted_loss4, predicted_loss5
    else: 
        predictions = predicted_loss1, predicted_loss3, predicted_loss4, predicted_loss5
    return predictions

def filter_by_list_of_genes(df, list_of_genes, exclude = False):
    if exclude:
        return df[~(df.gene_id1.isin(list_of_genes)|df.gene_id2.isin(list_of_genes))].reset_index(drop = True)
    else: 
        return df[df.gene_id1.isin(list_of_genes)|df.gene_id2.isin(list_of_genes)].reset_index(drop = True)

def obtain_bin_edges_and_target(df_int, df_con, specie):
    target_loss1 = calc_target_loss1(df_int, df_con)
    bin_edges_loss3 =  calc_bin_edges_loss3_4_5(df_int.area_of_the_matrix) 
    target_loss3 = calc_target_loss3_4_5(df_int.area_of_the_matrix, bin_edges_loss3)  #target_loss3 sums to 1
    bin_edges_loss4 = calc_bin_edges_loss3_4_5(df_con.area_of_the_matrix)
    target_loss4 =  calc_target_loss3_4_5(df_con.area_of_the_matrix, bin_edges_loss4)  #target_loss4 sums to 1
    bin_edges_loss5 = calc_bin_edges_loss3_4_5(df_int.area_of_the_interaction)
    target_loss5 =  calc_target_loss3_4_5(df_int.area_of_the_interaction, bin_edges_loss5) #target_loss5 sums to 1
    bin_edges = bin_edges_loss3, bin_edges_loss4, bin_edges_loss5
    if specie == 'human':
        target_loss2 = calc_target_loss2(df_int, df_con)
        targets = target_loss1, target_loss2, target_loss3, target_loss4, target_loss5
    else:
        targets = target_loss1, target_loss3, target_loss4, target_loss5
    return bin_edges, targets

def get_list_genes(df, unique = True):
    if unique:
        gene1 = df.gene_id1.unique() 
        gene2 = df.gene_id2.unique() 
        out = list(set(gene1) & set(gene2))
    else:
        gene1 = df.gene_id1
        gene2 = df.gene_id1
        out = list(np.concatenate((gene1, gene2), axis=0))
    return out

def find_the_best_couple(x, cand):
    sums = cand['difference'] + x.difference
    sums = sums.abs().sort_values(ascending = True)
    cand = cand.loc[sums.index]
    return cand.iloc[0].gene_id

def calculate_potential_loss(x, df, already_sampled, targets, bin_edges):
    gene1 = x.gene_id
    gene2 = x.best_gene_to_sample_with
    df_already_sampled= filter_by_list_of_genes(df, already_sampled) #work if already_sampled is a list
    df_to_sample = filter_by_list_of_genes(df, already_sampled, exclude = True)
    if df_already_sampled is None:
        assert df_to_sample.shape[0] == df.shape[0]
    else:
        assert df_to_sample.shape[0] + df_already_sampled.shape[0] == df.shape[0]
    subset = filter_by_list_of_genes(df_to_sample, [gene1, gene2])
    total_df = pd.concat([subset, df_already_sampled], axis = 0).reset_index(drop = True)
    df_int, df_con = split_df_int_con(total_df)
    try: #if this doesnt work, then df_int or df_con are empty, and I don t want this. So I penalize these candidates with a loss of 1000000
        assert len(df_int.species.value_counts()) == len(df_con.species.value_counts()) == 1
        specie = df_int.species.value_counts().index[0]
        predictions = calc_predictions_loss(df_int, df_con, specie, bin_edges)
        return calc_loss(targets, predictions, specie) 
    except:
        return 1000000 

def sampling_from_df_hub(df_hub, df_subset, gene_pairs_training, targets, bin_edges, take_into_account_the_first, random = False):
    """
    It samples two genes from df hub. df_subset can contain human or mouse, not both
    """
    df_hub = sort_df_hub(df_hub) # Sort df_hub per max(n_of_interactors, n_of_non_interactors)
    candidates = df_hub.head(take_into_account_the_first)
    candidates['difference'] = candidates['n_interactors'] - candidates['n_not_interactors']
    candidates['best_gene_to_sample_with'] = candidates.apply(find_the_best_couple, cand = candidates.copy(), axis = 1)
    if random:
        idx = np.random.randint(0,take_into_account_the_first) 
        gene_pair_to_sample = [candidates.iloc[idx].gene_id, candidates.iloc[idx].best_gene_to_sample_with]
    else:
        candidates['loss'] = candidates.apply(calculate_potential_loss, args=(df_subset, gene_pairs_training, targets, bin_edges), axis = 1)
        gene_pair_to_sample = [candidates.sort_values('loss').iloc[0].gene_id, candidates.sort_values('loss').iloc[0].best_gene_to_sample_with]
    rows_for_training = filter_by_list_of_genes(df_subset, list_of_genes = gene_pair_to_sample)

    df_hub = update_df_hub(df_hub, rows_for_training) #update df_hub (in order to update df_hub, remember to drop the gene picked and to update with -1 the n_interactors and n_not_interactors columns of the genes that are in pair with the gene picked. Remember, if n_interactors and n_not_interactors are 0 in a row, then I can drop this row).
    gene_pairs_training_updated = gene_pairs_training + list(rows_for_training.couples) #update gene_pairs_training
   
    return df_hub, gene_pairs_training_updated

def update_df_hub(df_hub, rows_for_training):
    rows_for_training_int, rows_for_training_con = split_df_int_con(rows_for_training)
    df_hub_diff = create_df_hub(rows_for_training_int, rows_for_training_con)
    df_hub = df_hub.merge(df_hub_diff, on = ['gene_id'], how = 'left')
    df_hub = df_hub.replace(np.nan, 0)
    df_hub['n_interactors'] = df_hub['n_interactors_x'] - df_hub['n_interactors_y']
    df_hub['n_not_interactors'] = df_hub['n_not_interactors_x'] - df_hub['n_not_interactors_y']
    return df_hub.drop(['n_interactors_x','n_interactors_y', 'n_not_interactors_x', 'n_not_interactors_y'], axis = 1).reset_index(drop = True)

def get_df(df_hub, dataset_data_dir, original_files_dir):
    filename = os.path.join(dataset_data_dir, 'df_train_test_val.csv')
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df_int = read_dataframe(os.path.join(original_files_dir, 'rise_paris_tr.new.mapped_interactions.tx_regions.txt'), columns_to_drop = ['Unnamed: 0','cdna_1','cdna_2']) #for the moment I don't need the sequences
        df_con = read_dataframe(os.path.join(original_files_dir, 'rise_paris_tr.controls.seq.txt'), columns_to_drop = ['Unnamed: 0', 'cdna_1','cdna_2']) #for the moment I don't need the sequences
        df_con = merge_features(df_int, df_con)
        df_int = create_features(df_int, df_hub = df_hub, df_interactors = True)
        df_con = create_features(df_con, df_hub = df_hub, df_interactors = False) #5 min to execute
        df = pd.concat([df_int, df_con]).sample(frac=1).reset_index(drop=True) #shuffle_rows and reset the index
        df['n_protein_codings_in_the_pair'] = df[['protein_coding_2', 'protein_coding_1']].apply(lambda x: sum(list(x)), axis = 1)
        assert df.shape[0] == df_int.shape[0] + df_con.shape[0]
        df.to_csv(filename, index = False)
    return df

def train_test_split(df, percentage_training = 0.7, random = False):
    df_human = df[df.species == 'human'].reset_index(drop = True)
    df_mouse = df[df.species == 'mouse'].reset_index(drop = True)
    print('HUMAN:')
    gene_pairs_training_human, gene_pairs_test_human = train_test_split_for_one_specie(df_human, percentage_training = percentage_training, random = random)
    print('\n')
    print('MOUSE:')
    gene_pairs_training_mouse, gene_pairs_test_mouse =  train_test_split_for_one_specie(df_mouse, percentage_training = percentage_training, random = random)
    gene_pairs_training = gene_pairs_training_human + gene_pairs_training_mouse
    gene_pairs_training = list(set(gene_pairs_training)) 
    gene_pairs_test = gene_pairs_test_human + gene_pairs_test_mouse
    gene_pairs_test = list(set(gene_pairs_test))
    assert len(gene_pairs_training) + len(gene_pairs_test) == len(df.couples.unique())
    return gene_pairs_training, gene_pairs_test

def train_test_split_for_one_specie(df_subset, percentage_training = 0.7, random = False):  
    start_time = time.time()
    #df_subset have to be 'human' or 'mouse', not both
    assert len(df_subset.species.value_counts()) == 1
    specie =  df_subset.species.value_counts().index[0]
    
    df_int, df_con = split_df_int_con(df_subset)
    
    df_hub = create_df_hub(df_int, df_con)

    bin_edges, targets = obtain_bin_edges_and_target(df_int, df_con, specie)
    
    gene_pairs_training = [] #Create a empty list of gene pairs samped 
    interrupt = False
    while (interrupt == False):
        df_hub, gene_pairs_training = sampling_from_df_hub(df_hub, df_subset, gene_pairs_training, targets, bin_edges, take_into_account_the_first = 50, random = random)
        gene_pairs_training = list(set(gene_pairs_training)) #get sure I have only unique values of gene_pairs_training
        interrupt = check_interruption(len(gene_pairs_training), len(df_subset.couples.unique()), percentage_training)
        if np.random.uniform()>0.85:
            print_progress(len(gene_pairs_training), len(df_subset.couples.unique()), start_time)

    gene_pairs_test = list(df_subset[~df_subset.couples.isin(gene_pairs_training)].couples.unique())
    
    assert len(gene_pairs_training) + len(gene_pairs_test) == len(df_subset.couples.unique())
    return gene_pairs_training, gene_pairs_test  

def print_progress(n_already_sample, n_total, start_time):
    actual_percentage = n_already_sample/n_total
    actual_percentage = np.round(actual_percentage, 4)*100
    total_time = np.round((time.time() - start_time)/60, 4)
    print("The actual training_set_percentage is around {}% (total time needed is around {} minutes)".format(actual_percentage, total_time))

def check_interruption(n_already_sampled, n_total, percentage_training, tolerance = 0.005):
    """
    higher the tolerance value is, less precise (with reagard to the percentage_training treshold fixed) is the training percentage of rna_rna_pairs sampled.
    """
    actual_percentage = n_already_sampled/n_total
        
    if actual_percentage>(percentage_training - tolerance):
        interrupt = True
        
    else:
        interrupt = False
    return interrupt

def obtain_list_of_unique_genes_from_pairs(gene_pairs):
    g12 = pd.Series(gene_pairs).str.split('_', expand = True)
    g1 = g12[0]
    g2 = g12[1]
    genes = list(g1) + list(g2)
    return list(dict.fromkeys(genes))
    
    
def plot_stats(df, train, test, val):
    """
    df can contain human or mouse, not both
    """
    
    gene_pairs_training = list(train.couples)
    gene_pairs_test = list(test.couples)
    gene_pairs_val = list(val.couples)
    
    venn_between_genes(gene_pairs_training, gene_pairs_test, gene_pairs_val)
    
    df_int, df_con = split_df_int_con(df)
    df_int_train, df_con_train = split_df_int_con(train)
    df_int_test, df_con_test = split_df_int_con(test)
    df_int_val, df_con_val = split_df_int_con(val)
    
    
    int_pair_perc = df_int.shape[0]/df.shape[0]
    int_pair_perc_train = df_int_train.shape[0]/train.shape[0]
    int_pair_perc_test = df_int_test.shape[0]/test.shape[0]
    int_pair_perc_val = df_int_val.shape[0]/val.shape[0]
    
    non_int_pair_perc = df_con.shape[0]/df.shape[0]
    non_int_pair_perc_train = df_con_train.shape[0]/train.shape[0]
    non_int_pair_perc_val = df_con_val.shape[0]/val.shape[0]
    non_int_pair_perc_test = df_con_test.shape[0]/test.shape[0]
    
    stacked_bar_chart_int_not_int([int_pair_perc, int_pair_perc_train, int_pair_perc_test, int_pair_perc_val],
                              [non_int_pair_perc, non_int_pair_perc_train, non_int_pair_perc_val, non_int_pair_perc_test],
                             'Percentages of interaction and non-interaction pairs in the three sets')
     
    pr_cod_pr_cod_int, pr_cod_no_pr_cod_int, no_pr_cod_no_pr_cod_int, pr_cod_pr_cod_con, pr_cod_no_pr_cod_con, no_pr_cod_no_pr_cod_con = calc_target_loss1(df_int, df_con)
    pr_cod_pr_cod_int_train, pr_cod_no_pr_cod_int_train, no_pr_cod_no_pr_cod_int_train, pr_cod_pr_cod_con_train, pr_cod_no_pr_cod_con_train, no_pr_cod_no_pr_cod_con_train = calc_target_loss1(df_int_train, df_con_train)
    pr_cod_pr_cod_int_test, pr_cod_no_pr_cod_int_test, no_pr_cod_no_pr_cod_int_test, pr_cod_pr_cod_con_test, pr_cod_no_pr_cod_con_test, no_pr_cod_no_pr_cod_con_test = calc_target_loss1(df_int_test, df_con_test)
    pr_cod_pr_cod_int_val, pr_cod_no_pr_cod_int_val, no_pr_cod_no_pr_cod_int_val, pr_cod_pr_cod_con_val, pr_cod_no_pr_cod_con_val, no_pr_cod_no_pr_cod_con_val = calc_target_loss1(df_int_val, df_con_val)
    
    stacked_bar_chart_biotype([pr_cod_pr_cod_int, pr_cod_pr_cod_int_train, pr_cod_pr_cod_int_test, pr_cod_pr_cod_int_val],
                      [pr_cod_no_pr_cod_int, pr_cod_no_pr_cod_int_train, pr_cod_no_pr_cod_int_test, pr_cod_no_pr_cod_int_val],
                      [no_pr_cod_no_pr_cod_int, no_pr_cod_no_pr_cod_int_train, no_pr_cod_no_pr_cod_int_test, no_pr_cod_no_pr_cod_int_val], 
                      'Percentages of transcript biotypes of the interaction pairs in the three sets')
    
    stacked_bar_chart_biotype([pr_cod_pr_cod_con, pr_cod_pr_cod_con_train, pr_cod_pr_cod_con_test, pr_cod_pr_cod_con_val],
                      [pr_cod_no_pr_cod_con, pr_cod_no_pr_cod_con_train, pr_cod_no_pr_cod_con_test, pr_cod_no_pr_cod_con_val],
                      [no_pr_cod_no_pr_cod_con, no_pr_cod_no_pr_cod_con_train, no_pr_cod_no_pr_cod_con_test, no_pr_cod_no_pr_cod_con_val], 
                      'Percentages of transcript biotypes of the non-interaction pairs in the three sets')
    
    assert len(df_int.species.value_counts()) == len(df_con.species.value_counts()) == 1
    specie = df_int.species.value_counts().index[0]
    
    #I want approximately the same percentage of the different cell lines in the three sets.
    if specie == 'human':
        Hek_int, Hela_highRNase_int, Hela_lowRNase_int, Hek_con, Hela_highRNase_con, Hela_lowRNase_con = calc_target_loss2(df_int, df_con)
        Hek_int_train, Hela_highRNase_int_train, Hela_lowRNase_int_train, Hek_con_train, Hela_highRNase_con_train, Hela_lowRNase_con_train = calc_target_loss2(df_int_train, df_con_train)
        Hek_int_test, Hela_highRNase_int_test, Hela_lowRNase_int_test, Hek_con_test, Hela_highRNase_con_test, Hela_lowRNase_con_test = calc_target_loss2(df_int_test, df_con_test)
        Hek_int_val, Hela_highRNase_int_val, Hela_lowRNase_int_val, Hek_con_val, Hela_highRNase_con_val, Hela_lowRNase_con_val = calc_target_loss2(df_int_test, df_con_test)
     
        stacked_bar_chart_cell_lines([Hek_int, Hek_int_train, Hek_int_test, Hek_int_val],
                      [Hela_highRNase_int, Hela_highRNase_int_train, Hela_highRNase_int_test, Hela_highRNase_int_val],
                      [Hela_lowRNase_int, Hela_lowRNase_int_train, Hela_lowRNase_int_test, Hela_lowRNase_int_val], 
                      'Percentages of transcript biotypes of the interaction pairs in the three sets')
        
        stacked_bar_chart_cell_lines([Hek_con, Hek_con_train, Hek_con_test, Hek_con_val],
                      [Hela_highRNase_con, Hela_highRNase_con_train, Hela_highRNase_con_test, Hela_highRNase_con_val],
                      [Hela_lowRNase_con, Hela_lowRNase_con_train, Hela_lowRNase_con_test, Hela_lowRNase_con_val], 
                      'Percentages of transcript biotypes of the non-interaction pairs in the three sets')
        
    else:
        assert df[['HEK293T','Hela(highRNase)','Hela(lowRNase)']].sum().sum() == 0
        print('The only cell line is mES')
    print('\n') 
        
    plt.title('Area of the Matrix Density')
    sns.kdeplot(filter_series(df['area_of_the_matrix'], 500000000),color = 'darkblue', linewidth = 1.5, label='original')
    sns.kdeplot(filter_series(train['area_of_the_matrix'], 500000000), color = 'green', linewidth = 1, label='train')
    sns.kdeplot(filter_series(test['area_of_the_matrix'], 500000000), color = 'red', linewidth = 1,  label='test')
    sns.kdeplot(filter_series(val['area_of_the_matrix'], 500000000), color = 'yellow', linewidth = 1,  label='val')
    x1,x2,y1,y2 = plt.axis()  
    plt.legend()
    plt.axis((x1,x2/2,y1,y2))
    plt.show()
    
    plt.title('Area of Interaction Density')
    sns.kdeplot(df_int['area_of_the_interaction'], color = 'darkblue', linewidth = 1.5, label='original')
    sns.kdeplot(df_int_train['area_of_the_interaction'], color = 'green', linewidth = 1, label='train')
    sns.kdeplot(df_int_test['area_of_the_interaction'], color = 'red', linewidth = 1,  label='test')
    sns.kdeplot(df_int_val['area_of_the_interaction'], color = 'yellow', linewidth = 1,  label='val')
    x1,x2,y1,y2 = plt.axis()  
    plt.legend()
    plt.axis((x1,3500,y1,y2))
    plt.show()

def stacked_bar_chart_biotype(pr_cod_pr_cod, pr_cod_no_pr_cod, no_pr_cod_no_pr_cod, title_string):
    labels = ['Original dataset', 'Training set', 'Test set', 'Validation set']
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, pr_cod_pr_cod, width, label='Protein coding-Protein coding')
    ax.bar(labels, pr_cod_no_pr_cod, width, bottom=pr_cod_pr_cod, label='Protein coding- No protein coding')
    ax.bar(labels, no_pr_cod_no_pr_cod, width, bottom=np.sum([np.array(pr_cod_no_pr_cod), np.array(pr_cod_pr_cod)], axis = 0), label='No protein coding- No protein coding')
    ax.set_ylabel('Percentage')
    ax.set_title(title_string)
    ax.legend()
    plt.show()
    
def stacked_bar_chart_cell_lines(Hek, Hela_highRNase, Hela_lowRNase, title_string):
    labels = ['Original dataset', 'Training set', 'Test set', 'Validation set']
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, Hek, width, label='HEK293T')
    ax.bar(labels, Hela_highRNase, width, bottom=Hek, label='Hela(highRNase)')
    ax.bar(labels, Hela_lowRNase, width, bottom=np.sum([np.array(Hek), np.array(Hela_highRNase)], axis = 0), label='Hela(lowRNase)')
    ax.set_ylabel('Percentage')
    ax.set_title(title_string)
    ax.legend()
    plt.show()

def stacked_bar_chart_int_not_int(int_perc, con_perc, title_string):
    labels = ['Original dataset', 'Training set', 'Test set', 'Validation set']
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, int_perc, width, label='Interaction pairs percentage')
    ax.bar(labels, con_perc, width, bottom=int_perc, label='Non-interaction pairs percentage')
    ax.set_ylabel('Percentage')
    ax.set_title(title_string)
    ax.legend()
    plt.show()
    
def filter_series(s, value):
    return s[s<value]

def create_or_load_train_test_val(df, save_path, random = False):
    """
    train-test-val percentages are around 70-20-10%
    """
    if random:
        file_names = ['gene_pairs_training_random.txt', 'gene_pairs_test_val_random.txt', 'gene_pairs_test_random.txt', 'gene_pairs_val_random.txt']
    else:
        file_names = ['gene_pairs_training.txt', 'gene_pairs_test_val.txt', 'gene_pairs_test.txt', 'gene_pairs_val.txt']

    file_name_training, file_name_test_val, file_name_test, file_name_val = file_names
    file_training = os.path.join(save_path, file_name_training)
    file_test_val = os.path.join(save_path, file_name_test_val)
    file_test = os.path.join(save_path, file_name_test)
    file_val = os.path.join(save_path, file_name_val)
    
    if (os.path.isfile(file_training))&(os.path.isfile(file_test_val)):
        with open(file_training, "rb") as fp:   # Unpickling
            gene_pairs_training = pickle.load(fp)

        with open(file_test_val, "rb") as fp:   # Unpickling
            gene_pairs_test_val = pickle.load(fp)
    else:
        gene_pairs_training, gene_pairs_test_val =  train_test_split(df, percentage_training = 0.7, random = random) #4,3 hours if not random
        with open(file_training, "wb") as fp:   #Pickling
            pickle.dump(gene_pairs_training, fp)
        with open(file_test_val, "wb") as fp:   #Pickling
            pickle.dump(gene_pairs_test_val, fp)
    
    train = df[df.couples.isin(gene_pairs_training)].reset_index(drop = True)
    test_val = df[df.couples.isin(gene_pairs_test_val)].reset_index(drop = True)
    assert df.merge(pd.DataFrame(gene_pairs_training + gene_pairs_test_val), left_on = 'couples', right_on = 0, how = 'inner').shape[0] == df.shape[0]
    
    if (os.path.isfile(file_test))&(os.path.isfile(file_val)):
        with open(file_test, "rb") as fp:   # Unpickling
            gene_pairs_test = pickle.load(fp)
        with open(file_val, "rb") as fp:   # Unpickling
            gene_pairs_val = pickle.load(fp)
    else:
        gene_pairs_test, gene_pairs_val =  train_test_split(test_val, percentage_training = 0.67, random = random) #0.67*0.3 = 0.2
        with open(file_test, "wb") as fp: #Pickling
            pickle.dump(gene_pairs_test, fp)

        with open(file_val, "wb") as fp: #Pickling
            pickle.dump(gene_pairs_val, fp)
            
    test = df[df.couples.isin(gene_pairs_test)].reset_index(drop = True)
    val = df[df.couples.isin(gene_pairs_val)].reset_index(drop = True)
    assert df.merge(pd.DataFrame(gene_pairs_test + gene_pairs_val), left_on = 'couples', right_on = 0, how = 'inner').shape[0] == test_val.shape[0]
    
    return train, test, val


def venn_between_genes(gene_pairs_training, gene_pairs_test, gene_pairs_val):
    genes_training = {g for g in obtain_list_of_unique_genes_from_pairs(gene_pairs_training)}
    genes_test = {g for g in obtain_list_of_unique_genes_from_pairs(gene_pairs_test)}
    genes_val = {g for g in obtain_list_of_unique_genes_from_pairs(gene_pairs_val)}
    
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    venn3([genes_training, genes_test, genes_val], set_labels = ('Training', 'Test', 'Validation'))
    plt.title('Genes in common between Training, Test and Validation sets \n')
    plt.show()