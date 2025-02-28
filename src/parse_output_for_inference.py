import os
import sys
import argparse
sys.path.insert(0, '..')
from util.inference_utils import associateRIMEpobability, ParseFasta, PlotByGene, format_output_table_bedpe

def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    parser.add_argument('--inference_dir', default='',
                        help='Path to the folder where you want to save the files for running RIME predictions')
    return parser


def main():

    pairs_prob_mean = associateRIMEpobability(temp_dir)
    output_table = format_output_table_bedpe(pairs_prob_mean)
    #output_table.to_csv(os.path.join(inference_dir,'output_table_withid.bedpe'), sep="\t", index=False)
    output_table["window_id1"] = output_table["window_id1"].str.split("_").str[0]
    output_table["window_id2"] = output_table["window_id2"].str.split("_").str[0]
    output_table.drop('id_pair', axis = 1).to_csv(os.path.join(inference_dir,'output_table.bedpe'), sep="\t", index=False)

    fasta_query_input = os.path.join(inference_dir, 'query.fa')
    fasta_target_input = os.path.join(inference_dir, 'target.fa')
    q_fa=ParseFasta(fasta_query_input)
    t_fa=ParseFasta(fasta_target_input)
    query_gene_names=q_fa.loc[:,"header"].to_list()
    target_gene_names=t_fa.loc[:,"header"].to_list()
    query_gene_lengths=q_fa.loc[:,"length"].to_list()
    target_gene_lengths=t_fa.loc[:,"length"].to_list()

    save_path_dir = os.path.join(inference_dir, 'plots')
    if os.path.isdir(save_path_dir) == False:
        os.mkdir(save_path_dir)


    for i, gene_x in enumerate(query_gene_names):
        for j, gene_y in enumerate(target_gene_names):
            save_path = os.path.join(save_path_dir, f'{gene_x}_{gene_y}.png')
            size_x, size_y = max(1, query_gene_lengths[i]//200), max(1, target_gene_lengths[j]//200)
            PlotByGene(pairs_prob_mean, gene_x, gene_y, save_path = save_path, sizes = (size_y, size_x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare data for inference', parents=[get_args_parser()])
    args = parser.parse_args()
    inference_dir = args.inference_dir
    temp_dir = os.path.join(inference_dir, 'temp')

    main()