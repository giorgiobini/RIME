import os
import sys
import argparse
sys.path.insert(0, '..')
from util.inference_utils import associateRIMEpobability, ParseFasta, PlotByGene

# import subprocess

def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    # parser.add_argument('--model_name', default='arch2_PSORALENtrained_PARISval0046',
    #                     help='Name of the model folder')
    parser.add_argument('--inference_dir', default='',
                        help='Path to the folder where you want to save the files for running RIME predictions')
    return parser



if __name__ == '__main__':
    #run me with: -> 
    #nohup python parse_output_for_inference.py --inference_dir=/data01/giorgio/RNARNA-NT/dataset/external_dataset/check_parse_fasta_class/ &> parse_output_for_inference.out &

    parser = argparse.ArgumentParser('Prepare data for inference', parents=[get_args_parser()])
    args = parser.parse_args()
    inference_dir = args.inference_dir
    temp_dir = os.path.join(inference_dir, 'temp')

    pairs_prob_mean = associateRIMEpobability(temp_dir)
    pairs_prob_mean.drop(['window_1', 'window_2'], axis =1).to_csv(os.path.join(inference_dir,'output_table.bedpe'), sep="\t", index=False)

    fasta_query_input = os.path.join(inference_dir, 'query.fa')
    fasta_target_input = os.path.join(inference_dir, 'target.fa')
    q_fa=ParseFasta(fasta_query_input)
    t_fa=ParseFasta(fasta_target_input)
    query_gene_names=q_fa.loc[:,"header"].to_list()
    target_gene_names=t_fa.loc[:,"header"].to_list()

    save_path_dir = os.path.join(inference_dir, 'plots')
    if os.path.isdir(save_path_dir) == False:
        os.mkdir(save_path_dir)

    for gene_x in query_gene_names:
        for gene_y in target_gene_names:
            save_path = os.path.join(save_path_dir, f'{gene_x}_{gene_y}.png')
            PlotByGene(pairs_prob_mean, gene_x, gene_y, save_path = save_path, sizes = (50, 20))


# ### GIORGIO SCRIPTS
# obj_rinet.LoadEmbedding()
# obj_rinet.InferProbability()

# ### FINE GIORGIO SCRIPTS

# obj_rinet.AssociateRInetProbability()
# #obj_rinet.PlotByGene("NM_001396408.1","PGLYRP3",[10,15])