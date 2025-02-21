# RIME
This repository provides instructions, including dependencies and scripts, for using RIME (RNA Interactions Model with Embeddings), a deep learning framework designed for predicting interactions between long RNA molecules. In particular, we provide the RIMEfull model, which was trained and validated using the complete Psoralen-based set. Starting from two input RNA sequences, RIMEfull prediction scores are calculated across 200x200 nucleotide windows with a 100-nucleotide step. If you only want to run RIMEfull on a single pair of RNA molecules, you can use the dedicated web service, available at https://tools.tartaglialab.com/rna_rna

<img src="RIMElogo.jpg">

If this source code is helpful for your research, please cite the following publication:

"Decoding RNA-RNA Interactions: The Role of Low-Complexity Repeats and a Deep Learning Framework for Sequence-Based Prediction"

Adriano Setti†, Giorgio Bini†, Valentino Maiorca, Flaminia Pellegrini, Gabriele Proietti, Dimitrios Miltiadis-Vrachnos, Alexandros Armaos, Julie Martone, Michele Monti, Giancarlo Ruocco, Emanuele Rodolà, Irene Bozzoni, Alessio Colantoni*, Gian Gaetano Tartaglia*

†Co-first authors
*Corresponding authors

https://www.biorxiv.org/content/10.1101/2025.02.16.638500v1

## 1. Environment setup
We recommend building a Python virtual environment with Anaconda. This repository enables running RIME with GPU acceleration. RIME was developed on a machine with CUDA version 525, CUDA toolkit version 12.0, and cuDNN version 8.8.1.3.

### 1.1 Install Miniconda  
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
export PATH="/home/username/miniconda/bin:$PATH"
```

### 1.2 Create and activate a new virtual environment, install the package, and other dependencies  

#### **Nucleotide Transformer embeddings (Required)**  
```
conda create -n download_embeddings python=3.10 ipython 
conda activate download_embeddings
conda install pandas=1.5.3 -y
conda install -c anaconda seaborn=0.12.2 -y
conda install pytorch torchvision -c pytorch -y
conda install -c anaconda scikit-learn=1.2.2 -y
pip install StrEnum==0.4.8
cd ./NT_dependencies
pip install .
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pytorch-gpu
```

#### **RIME (Required)**  
```
conda create -n rime --file spec-file-rime.txt
conda activate rime
pip install -r other-requirements-rime.txt
```

### 1.3 Install BEDtools
The inference scripts require the BEDTools suite to be installed on your system. You can install BEDTools by following the instructions provided here: https://bedtools.readthedocs.io/en/latest/content/installation.html. The path to the BEDTools binary (referred to as `/path_to_bedtools/bin/bedtools`) will be needed as input in the initial step of the procedure.

## 2. Model  
To run RIME, a model should be present in the `./checkpoints` folder:

```
checkpoints
│ 
└── RIMEfull
```

You can download the RIMEfull model folder from this link: [Insert Link Here]  


## 3. Inference

### Required Files
The procedure requires two input FASTA files: one for query RNA sequences (hereby called `query.fa`) and one for target RNA sequences (hereby called `target.fa`). Please note that U and T characters are considered equivalent. Both files can contain multiple sequences and must be placed within the same directory, hereby called `/path/to/input/files/`. Inference will be performed on all possible query-target pairs. The procedure also requires an output directory, hereby called `/path/to/output/files/`

### Running Inference  
Run the following commands from the `./src` directory:

1. **Activate Conda environment**  
   ```
   conda activate download_embeddings
   ```
2. **Parse FASTA files**  
This script generates the 200x200 windows and prepares the input files for embedding extraction
   ```
   python parse_fasta_for_inference.py --bin_bedtools=/path_to_bedtools/bin/bedtools \
   --inference_dir=/path/to/output/files/ \
   --input_dir=/path/to/input/files/ \
   --fasta_query_name=query.fa \
   --fasta_target_name=target.fa \
   --analysis_name=temp
   ```
3. **Download embeddings**  
   ```
   python download_embeddings.py --batch_size=1 \
   --analysis_dir=/path/to/output/files/temp 
   ```
4. **Activate RIME environment**  
   ```
   conda activate rime
   ```
5. **Run inference**  
   ```
   python run_inference.py --analysis_dir=/path/to/output/files/temp \
   --model_name=RIMEfull
   ```
6. **Parse output**  
   ```
   python parse_output_for_inference.py \
   --inference_dir=/path/to/output/files/
   ```

### Output:  
After running these steps, you will find the following inside `/path/to/output/files/ `:  
- **`output_table.bedpe`**  A BEDPE file containing RIMEfull prediction scores for all 200×200 windows generated from every target-query pair.
- **`plots/`**  A directory containing PNG files, each representing a heatmap of RIMEfull scores for a specific target-query pair.


### Note:
- The **25B multi-species NT model** (~10GB) will be **downloaded automatically** the first time you run `download_embeddings.py`. It will be stored in `./NT_dependencies/checkpoints/`.  
