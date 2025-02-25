import os
import sys
import argparse
sys.path.insert(0, '..')
from util.inference_utils import RIME_inference

# import subprocess


### CLASS AND FUNCTIONS 

def get_args_parser():
    parser = argparse.ArgumentParser('Set model args', add_help=False)
    parser.add_argument('--bin_bedtools', default='/home/giorgio/bedtools2/bin/bedtools',
                        help='Path to the bedtools2 folder')
    parser.add_argument('--inference_dir', default='',
                        help='Path to the folder where you want to save the files for running RIME predictions')
    parser.add_argument('--input_dir', default='',
                        help='Path to the folder where there is query and tartet fasta')
    parser.add_argument('--fasta_query_name', default='',
                        help='Name of the fasta query file, e.g. TINCR.fasta')
    parser.add_argument('--fasta_target_name', default='',
                        help='Name of the fasta target file, e.g PGLYRP3.fa')
    parser.add_argument('--analysis_name', default='',
                        help='Name of the analysis, e.g TINCR_PGLYRP3')

    parser.add_argument('--size_window', default=200, type=int)
    parser.add_argument('--step_window', default=100, type=int)
    parser.add_argument('--length_max_embedding', default=5970, type=int)
    return parser


def main():
    ### CODE
    obj_rinet = RIME_inference(bin_bedtools, fasta_query_input,fasta_target_input,dir_out,name_analysis=name_analysis)
    wind_df=obj_rinet.Windows(size_window=size_window,step=step_window,mode="generate")
    emb_df=obj_rinet.Embedding(mode="generate",length_max_embedding=length_max_embedding,step=step_embedding)
    obj_rinet.AssembleQueryTargetRanges()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare data for inference', parents=[get_args_parser()])
    args = parser.parse_args()
    fasta_query_input = os.path.join(args.input_dir, args.fasta_query_name)
    fasta_target_input = os.path.join(args.input_dir, args.fasta_target_name)
    dir_out = args.inference_dir
    name_analysis = args.analysis_name
    bin_bedtools = args.bin_bedtools

    size_window = args.size_window
    step_window = args.step_window
    length_max_embedding = args.length_max_embedding
    step_embedding=int(length_max_embedding/2)
    main()