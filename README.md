# RNA-RNA interactions classifier
This repo bla bla

## 1. Environment setup 
We recommend you to build a python virtual environment with Anaconda.

#### 1.1 Install miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
export PATH="/home/username/miniconda/bin:$PATH"
```

#### 1.2 Create and activate a new virtual environment

```
conda create --name rnarna
conda activate rnarna

cd NT_dependencies
python3 -m pip install --editable .
```

(Optional)
```
conda install -c conda-forge jupyterlab
```

(Optional)
```
conda create --name cdhit
conda activate cdhit
conda install -c bioconda cd-hit=4.8.1
```

(Optional - only for NT)
```
cd NT_dependencies
python3 -m pip install --editable .
pip install tensorboardX
pip install tensorboard
pip install scikit-learn >= 0.22.2
pip install seqeval
pip install pyahocorasick
pip install scipy
pip install statsmodels
pip install biopython
pip install pandas
pip install pybedtools
pip install sentencepiece==0.1.91
conda install -c conda-forge tokenizers
conda install pytorch
```

#### 1.3 Install the package and other requirements

??

(Required)

```
cd RNARNA
cd DNABERT_dependencies
python3 -m pip install --editable .
cd ..
pip install -r requirements.txt
conda install munch
conda install -c conda-forge matplotlib-venn
python -m pip install --no-cache-dir ortools
``` 
 
## 2. Data
We need the data to be structured as the example below.

```
dataset
├── original_files
│   └── hub.table.paris.txt
│   └── rise_paris_tr.controls.seq.txt
│   └── rise_paris_tr.new.mapped_interactions
│ 
├── pre_trained_DNABERT
│   └── 6-new-12w-0
│       └──config.json
│       └──pytorch_model.bin
│       └──special_tokens_map.json
│       └──tokenizer_config.json
│       └──vocab.txt
├── annotation_files
├── processed_files
└── rna_rna_pairs
```

You can download the original_files folder from this link (put a link). If you don't need to train the model you can avoid to download the original_files and keep the directory empty.

You can download the 6-new-12w-0 folder from this [link](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view).

## 3. Train your model
Skip this section if you only need inference.

Run these scripts from the src directory in the following order:

- train_test_val.ipynb
- Preprocessing.ipynb (if you want to map the overlapping bounding boxes to a common bounding box which include them)
- python create_fasta_query_for_secondary_structure.py
- cd ../UFold_dependencies/
- python run_ufold_predictions.py
- cd ../src/
- python dot_bracket_preprocessing.py
- Filter_data.ipynb
- create_seq_for_rnabert.ipynb
- conda activate rnabert
- train_rnabert.sh
- conda activate rnarna
- data_augmentation.ipynb (OPTIONAL, only to see if the dataloader works)
- Statistics_on_distribution_dataset.ipynb (OPTIONAL, only to see the expeted values of cds-cds, etc..)
- Decide_hyperparameters_dataloader.ipynb
- train.ipynb

(OPIONAL mmseq2)
- create_fasta_query_for_mmseq2.ipynb
- conda activate mmseq2
- cd ./dataset/processed_files/mmseq2/
- mmseqs createdb fasta_mmseq2.txt DB
- mmseqs cluster DB DB_clu tmp --cov-mode 0 -c 0.8 --min-seq-id 0.75
- mmseqs createseqfiledb DB DB_clu DB_clu_seq
- mmseqs result2flat DB DB DB_clu_seq DB_clu_seq.fasta
- conda activate rnarna
- cd /RNARNA/src/
- process_mmseq2_results.ipynb
- analyze_clustering.ipynb

(OPIONAL cd-hit)
- create_fasta_query_for_cdhit.ipynb
- conda activate cdhit
- cd ./dataset/processed_files/cdhit/
- cd-hit-est -i fasta_cdhit.txt -o cdhit_clustering -c 0.8 -n 5 -T 10 -r 0 -d 0
- cd-hit-est -i fasta_cdhit_fl.txt -o cdhit_clustering_fl -G 0 -c 0.8 -aL 0.25 -aS 0.25 -T 10 -r 0 -d 0
- cd-hit-est -i fasta_cdhit_fl.txt -o cdhit_clustering_fl2 -G 0 -c 0.8 -aL 0.5 -aS 0.5 -T 10 -r 0 -d 0
- process_cdhit_results.ipynb
- analyze_clustering.ipynb

## 4. Inference
Put your files inside the directory dataset/external_dataset/your_folder/
You must have these files inside your_folder:

```
genes.fa 

>ENSFAKE001
CGUUUCGCUAAACUCUG
>ENSFAKE002
UCGCGAGGCGCAACGGCGCCGACCGAGUGUAGGC
>ENSFAKE003
GUGAACGUCGCGAUAGGCGGAACAA
>ENSFAKE004
AGUAACAACGCUAGGUGCGAGUGUCGUC


pairs.txt

ENSFAKE001_ENSFAKE002
ENSFAKE001_ENSFAKE003
ENSFAKE003_ENSFAKE004
```

Run these scripts from the src directory in the following order:
- python create_fasta_query_for_secondary_structure.py --dataset='inference' --input_dir='/data01/gbini/projects/DETR/dataset/external_dataset/pulldown' --results_dir='/data01/gbini/projects/DETR/dataset/external_dataset/pulldown/data'
- cd ../UFold_dependencies/
- python run_ufold_predictions.py --files_dir='/data01/gbini/projects/DETR/dataset/external_dataset/pulldown/data' --results_dir='/data01/gbini/projects/DETR/dataset/external_dataset/pulldown/results'
- cd ../src/
- python dot_bracket_preprocessing.py  --files_dir='/data01/gbini/projects/DETR/dataset/external_dataset/pulldown/results' --results_dir='/data01/gbini/projects/DETR/dataset/external_dataset/pulldown'
- run_full_length.ipynb




