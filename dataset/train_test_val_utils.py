import pandas as pd
import os
import time
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns
import pickle

def obtain_list_of_unique_genes_from_pairs(gene_pairs):
    g12 = pd.Series(gene_pairs).str.split('_', expand = True)
    g1 = g12[0]
    g2 = g12[1]
    genes = list(g1) + list(g2)
    return list(dict.fromkeys(genes))

def venn_between_genes(gene_pairs_training, gene_pairs_test, gene_pairs_val):
    genes_training = {g for g in obtain_list_of_unique_genes_from_pairs(gene_pairs_training)}
    genes_test = {g for g in obtain_list_of_unique_genes_from_pairs(gene_pairs_test)}
    genes_val = {g for g in obtain_list_of_unique_genes_from_pairs(gene_pairs_val)}
    
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    venn3([genes_training, genes_test, genes_val], set_labels = ('Training', 'Test', 'Validation'))
    plt.title('Genes in common between Training, Test and Validation sets \n')
    plt.show()
    
def check_interruption(n_already_sampled, n_total, percentage_training, tolerance = 0.005, perc_print = 0.005):
    """
    higher the tolerance value is, less precise (with reagard to the percentage_training treshold fixed) is the training percentage of rna_rna_pairs sampled.
    """
    actual_percentage = n_already_sampled/n_total
    progress = actual_percentage/(1-percentage_training)
    if np.random.uniform()<perc_print:
        print(f'progress around {np.round(progress*100, 2)}%')
        
    if actual_percentage>(percentage_training - tolerance):
        interrupt = True
        
    else:
        interrupt = False
    return interrupt

def sample_gene(list_of_genes):
    random.shuffle(list_of_genes)
    return list_of_genes[0], list_of_genes[1:]

def sample_one_gene(df, gene_sampled):
    condition = (df.gene1 == gene_sampled)|(df.gene2 == gene_sampled)|\
                (df.gene2_neg == gene_sampled)|(df.gene2_neg == gene_sampled)
    return list(set(df[condition].positive)), df[~condition]
    

def train_test_split_from_df_pairs(df_subset, percentage_training = 0.7, seed = 123): 
    random.seed(seed)
    percentage_test = 1-percentage_training
    all_possible_pairs = set(df_subset.positive)
    list_of_genes = list(set(df_subset.gene1).union(set(df_subset.gene2)))
    gene_pairs_test = [] #Create a empty list of gene pairs samped 
    interrupt = False
    while (interrupt == False):
        gene_sampled, list_of_genes = sample_gene(list_of_genes)
        #campiono il gene, prendo tutte le coppie positive che riguardano il gene campionato,
        #e campiono di conseguenza anche i rispettivi negativi. 
        #Campiono anche tutte le coppie negative che riguardano il gene campionato,
        #e campiono di conseguenza anche i rispettivi positivi.
        positive_set_sampled, df_subset = sample_one_gene(df_subset, gene_sampled)
        gene_pairs_test += positive_set_sampled
        interrupt = check_interruption(len(gene_pairs_test), len(all_possible_pairs), percentage_test)
    
    gene_pairs_training = set(all_possible_pairs) - set(gene_pairs_test)
    return gene_pairs_training, gene_pairs_test  


def create_or_load_train_test_val(df, save_path):
    
    start_time = time.time()
    file_names = ['gene_pairs_training.txt', 'gene_pairs_test.txt', 'gene_pairs_val.txt']
    file_name_training, file_name_test, file_name_val = file_names
    file_training = os.path.join(save_path, file_name_training)
    file_test = os.path.join(save_path, file_name_test)
    file_val = os.path.join(save_path, file_name_val)
    
    if (os.path.isfile(file_training)) & (os.path.isfile(file_test)) & (os.path.isfile(file_val)):
        with open(file_training, "rb") as fp:   # Unpickling
            gene_pairs_training = pickle.load(fp)

        with open(file_test, "rb") as fp:   # Unpickling
            gene_pairs_test= pickle.load(fp)

        with open(file_val, "rb") as fp:   # Unpickling
            gene_pairs_val= pickle.load(fp)
    else:
        gene_pairs_training_val, gene_pairs_test = train_test_split_from_df_pairs(df, percentage_training = 0.85)
        gene_pairs_training, gene_pairs_val = train_test_split_from_df_pairs(df[df.positive.isin(gene_pairs_training_val)], percentage_training = 0.83)
        gene_pairs_training, gene_pairs_test, gene_pairs_val = list(gene_pairs_training), list(gene_pairs_test), list(gene_pairs_val)
        
        # add negative pairs
        gene_pairs_training += list(set(df[df.positive.isin(gene_pairs_training)].negative))
        gene_pairs_test += list(set(df[df.positive.isin(gene_pairs_test)].negative))
        gene_pairs_val += list(set(df[df.positive.isin(gene_pairs_val)].negative))
        
        with open(file_training, "wb") as fp:   #Pickling
            pickle.dump(gene_pairs_training, fp)

        with open(file_test, "wb") as fp:   #Pickling
            pickle.dump(gene_pairs_test, fp)

        with open(file_val, "wb") as fp:   #Pickling
            pickle.dump(gene_pairs_val, fp)
            
    print(f"Total time: {(time.time()-start_time)/60} minutes")
            
    return gene_pairs_training, gene_pairs_test, gene_pairs_val

# - - - - - - - - - - - - - - - - - - - - - - - - - - PLOTS - - - - - - - - - - - - - - - - - - - - - - - - - - 

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
    
def split_df_int_con(df):
    return df[df.interacting == True].reset_index(drop = True), df[df.interacting == False].reset_index(drop = True)

def calc_target_loss1(df_int, df_con):
    pr_cod_pr_cod_int = df_int[df_int.n_protein_codings_in_the_pair == 2].shape[0]/df_int.shape[0] #percentage of protein_coding-protein_coding in the interaction pair 
    pr_cod_no_pr_cod_int = df_int[df_int.n_protein_codings_in_the_pair == 1].shape[0]/df_int.shape[0] #percentage of protein_coding-non_protein_coding in the interaction pair
    no_pr_cod_no_pr_cod_int = df_int[df_int.n_protein_codings_in_the_pair == 0].shape[0]/df_int.shape[0] #percentage of non_protein_coding-non_protein_coding in the interaction pair is
    pr_cod_pr_cod_con = df_con[df_con.n_protein_codings_in_the_pair == 2].shape[0]/df_con.shape[0] #percentage of protein_coding-protein_coding in the non interaction pair
    pr_cod_no_pr_cod_con = df_con[df_con.n_protein_codings_in_the_pair == 1].shape[0]/df_con.shape[0] #percentage of protein_coding-non_protein_coding in the non interaction pair
    no_pr_cod_no_pr_cod_con = df_con[df_con.n_protein_codings_in_the_pair == 0].shape[0]/df_con.shape[0] #percentage of non_protein_coding-non_protein_coding in the non interaction pair
    return np.asarray([pr_cod_pr_cod_int, pr_cod_no_pr_cod_int, no_pr_cod_no_pr_cod_int, pr_cod_pr_cod_con, pr_cod_no_pr_cod_con, no_pr_cod_no_pr_cod_con])

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

def plot_stats(df, gene_pairs_training, gene_pairs_test, gene_pairs_val, limit_plot = 100000000):
    df['n_protein_codings_in_the_pair'] = df[['protein_coding_2', 'protein_coding_1']].apply(lambda x: sum(list(x)), axis = 1)

    df['area_of_the_matrix'] =  df.length_1*df.length_2

    df['area_of_the_interaction'] = df.w*df.h


    train = df[df.couples.isin(gene_pairs_training)]
    test = df[df.couples.isin(gene_pairs_test)]
    val = df[df.couples.isin(gene_pairs_val)]

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
    
    plt.title('Area of the Matrix Density')
    sns.kdeplot(filter_series(df['area_of_the_matrix'], limit_plot),color = 'darkblue', linewidth = 1.5, label='original')
    sns.kdeplot(filter_series(train['area_of_the_matrix'], limit_plot), color = 'green', linewidth = 1, label='train')
    sns.kdeplot(filter_series(test['area_of_the_matrix'], limit_plot), color = 'red', linewidth = 1,  label='test')
    sns.kdeplot(filter_series(val['area_of_the_matrix'], limit_plot), color = 'yellow', linewidth = 1,  label='val')
    x1,x2,y1,y2 = plt.axis()  
    plt.legend()
    plt.axis((x1,x2/2,y1,y2))
    plt.show()

    plt.title('Area of the Matrix (Training) Interactive VS Not-Interactive pairs')
    sns.kdeplot(filter_series(df_int_train['area_of_the_matrix'], limit_plot), color = 'green', linewidth = 1, label='interactive')
    sns.kdeplot(filter_series(df_con_train['area_of_the_matrix'], limit_plot), color = 'black', linewidth = 1, label='not interactive')
    x1,x2,y1,y2 = plt.axis()  
    plt.legend()
    plt.axis((x1,x2/2,y1,y2))
    plt.show()

    plt.title('Length gene (Training) Interactive VS Not-Interactive pairs')
    l_g1 = pd.Series(list(df_int_train.length_1) + list(df_int_train.length_2))
    l_g2 = pd.Series(list(df_con_train.length_1) + list(df_con_train.length_2))
    sns.kdeplot(filter_series(l_g1, 30000), color = 'darkblue', linewidth = 1, label='length interactive')
    sns.kdeplot(filter_series(l_g2, 30000), color = 'green', linewidth = 1, label='length not interactive')
    x1,x2,y1,y2 = plt.axis()  
    plt.legend()
    plt.axis((x1,x2/2,y1,y2))
    plt.show()

    plt.title('Length gene1 VS Length gene2 (Training) Interactive VS Not-Interactive pairs')
    sns.kdeplot(filter_series(df_int_train.length_1, 30000), color = 'darkblue', linewidth = 1, label='gene1 interactive')
    sns.kdeplot(filter_series(df_int_train.length_2, 30000), color = 'green', linewidth = 1, label='gene2 interactive')
    sns.kdeplot(filter_series(df_con_train.length_1, 30000), color = 'red', linewidth = 1, label='gene1 not interactive')
    sns.kdeplot(filter_series(df_con_train.length_2, 30000), color = 'yellow', linewidth = 1, label='gene2 not interactive')
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