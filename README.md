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
```

(Optional)
```
conda install -c conda-forge jupyterlab
```

#### 1.3 Install the package and other requirements

??

(Required)

```
pip install -U jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cd NT_dependencies
git clone https://github.com/instadeepai/nucleotide-transformer.git nt
mv nt/* . 
mv nt/.* .
rmdir nt
cp mypretrained.py nucleotide_transformer/
pip install .
cd ..
pip install -r requirements.txt
conda install pytorch
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
├── annotation_files
├── processed_files
└── rna_rna_pairs
```

You can download the original_files folder from this link (put a link). If you don't need to train the model you can avoid to download the original_files and keep the directory empty.

## 3. Train your model
Skip this section if you only need inference.

Run these scripts from the src directory in the following order:
-
-

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
-
-