#!/usr/bin/env python3
import subprocess
import argparse
import os
import shutil
import warnings

def run_command(cmd, env):
    """
    Run a shell command inside a conda environment using conda run,
    making sure PYTHONPATH includes the repo root so 'util' is found.
    """
    repo_root = os.path.dirname(__file__)

    subprocess.run(
        f"conda run -n {env} bash -c 'export PYTHONPATH={repo_root}:$PYTHONPATH && {cmd}'",
        shell=True,
        check=True,
        cwd=repo_root
    )

def run_pipeline(input_dir, query, target, output_dir, model="RIMEfull", bedtools_path="/path_to_bedtools/bin/bedtools"):
    temp_dir = os.path.join(output_dir, "temp")

    warnings.simplefilter('ignore', FutureWarning)

    # Step 1: Parse FASTA
    run_command(
        f"python src/parse_fasta_for_inference.py "
        f"--bin_bedtools={bedtools_path} "
        f"--inference_dir={output_dir} "
        f"--input_dir={input_dir} "
        f"--fasta_query_name={query} "
        f"--fasta_target_name={target} "
        f"--analysis_name=temp",
        env="download_embeddings"
    )

    # Step 2: Download embeddings
    run_command(
        f"python src/download_embeddings.py "
        f"--batch_size=1 "
        f"--analysis_dir={temp_dir}",
        env="download_embeddings"
    )

    # Step 3: Run inference
    run_command(
        f"python src/run_inference.py "
        f"--analysis_dir={temp_dir} "
        f"--model_name={model}",
        env="rime"
    )

    # Step 4: Parse output
    run_command(
        f"python src/parse_output_for_inference.py "
        f"--input_dir={input_dir} "
        f"--inference_dir={output_dir} "
        f"--fasta_query_name={query} "
        f"--fasta_target_name={target}",
        env="rime"
    )

    # Step 5: Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Deleted temporary directory: {temp_dir}")

    print(f"Inference complete. Results available in {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run RIME inference pipeline.")
    parser.add_argument("--input_dir", required=True, help="Directory with input FASTA files")
    parser.add_argument("--query", required=True, help="Query FASTA file name")
    parser.add_argument("--target", required=True, help="Target FASTA file name")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--model", default="RIMEfull", help="Model name (default: RIMEfull)")
    parser.add_argument("--bedtools_path", default="/path_to_bedtools/bin/bedtools", help="Path to bedtools executable")

    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        query=args.query,
        target=args.target,
        output_dir=args.output_dir,
        model=args.model,
        bedtools_path=args.bedtools_path
    )
