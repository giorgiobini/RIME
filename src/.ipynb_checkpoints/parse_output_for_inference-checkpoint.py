import os
import sys
import argparse
sys.path.insert(0, '..')
from util.inference_utils import associateRIMEpobability, ParseFasta

# import subprocess

def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    # parser.add_argument('--model_name', default='arch2_PSORALENtrained_PARISval0046',
    #                     help='Name of the model folder')
    parser.add_argument('--output_file_dir', default='',
                        help='Path to the folder where you want to save the files for running RIME predictions')
    parser.add_argument('--fasta_path', default='',
                        help='Path to the folder where there is query and tartet fasta')
    parser.add_argument('--fasta_query_name', default='',
                        help='Name of the fasta query file, e.g. TINCR.fasta')
    parser.add_argument('--fasta_target_name', default='',
                        help='Name of the fasta target file, e.g PGLYRP3.fa')
    parser.add_argument('--name_analysis', default='',
                        help='Name of the analysis, e.g TINCR_PGLYRP3')

    parser.add_argument('--size_window', default=200, type=int)
    parser.add_argument('--step_window', default=100, type=int)
    parser.add_argument('--length_max_embedding', default=5970, type=int)
    return parser



if __name__ == '__main__':
    #run me with: -> 
    #nohup python parse_fasta_for_run_inference.py --output_file_dir=/data01/giorgio/RNARNA-NT/dataset/external_dataset/check_parse_fasta_class/ --fasta_path=/data01/giorgio/RNARNA-NT/dataset/external_dataset/check_parse_fasta_class/ --fasta_query_name=query.fa --fasta_target_name=target.fa --name_analysis=prova&> parse_fasta_for_run_inference.out &

    parser = argparse.ArgumentParser('Prepare data for inference', parents=[get_args_parser()])
    args = parser.parse_args()
    fasta_query_input = os.path.join(args.fasta_path, args.fasta_query_name)
    fasta_target_input = os.path.join(args.fasta_path, args.fasta_query_name)
    dir_out = args.output_file_dir
    name_analysis = args.name_analysis

    size_window = args.size_window
    step_window = args.step_window
    length_max_embedding = args.length_max_embedding
    step_embedding=int(length_max_embedding/2)
    associateRIMEpobability()




# ### GIORGIO SCRIPTS
# obj_rinet.LoadEmbedding()
# obj_rinet.InferProbability()

# ### FINE GIORGIO SCRIPTS

# obj_rinet.AssociateRInetProbability()
# #obj_rinet.PlotByGene("NM_001396408.1","PGLYRP3",[10,15])