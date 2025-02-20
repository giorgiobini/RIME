# RIME
This repository provides instructions, including dependencies and scripts, for using RIME (RNA Interactions Model with Embeddings), a deep learning framework designed for predicting RNA-RNA interactions. If you only want to test a single RNA-RNA interaction we released the webserver at https://tools.tartaglialab.com/rna_rna

<img src="RIMElogo.jpg">

If this source code is helpful for your research please cite the following publication:

"Decoding RNA-RNA Interactions: The Role of Low-Complexity Repeats and a Deep Learning Framework for Sequence-Based Prediction"

Adriano Setti*†, Giorgio Bini*†, Valentino Maiorca, Flaminia Pellegrini, Gabriele Proietti, Dimitrios Miltiadis-Vrachnos, Alexandros Armaos, Julie Martone, Michele Monti, Giancarlo Ruocco, Emanuele Rodolà, Irene Bozzoni, Alessio Colantoni‡, Gian Gaetano Tartaglia‡

*† Co-first authors
‡ Co-last authors

https://www.biorxiv.org/content/10.1101/2025.02.16.638500v1

## 1. Environment setup  
We recommend you to build a Python virtual environment with Anaconda.  
The machine has CUDA version **525**, toolkit **CUDA 12.0**, and **cuDNN 8.8.1.3**.  

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

### 1.3 Install bedtools2  
The inference scripts require **bedtools2** installed on your machine.  

## 2. Model  
We need the data to be structured as shown below:  

```
checkpoints
│ 
└── RIMEfull
```

You can download the **RIMEfull** folder from this link: [Insert Link Here]  


## 3. Inference
Put these files inside the directory:  
```
./dataset/external_dataset/your_folder/
```
### Required Files:
#### `query.fa`
```
>ENSFAKE001
CGUUUCGCUAAACUCUG
>ENSFAKE002
UCGCGAGGCGCAACGGCGCCGACCGAGUGUAGGC
```
#### `target.fa`
```
>ENSFAKE003
GUGAACGUCGCGAUAGGCGGAACAA
>ENSFAKE004
AGUAACAACGCUAGGUGCGAGUGUCGUC
```
### Running Inference  
Run the following scripts from the `./src` directory in order:

1. **Activate Conda environment**  
   ```
   conda activate download_embeddings
   ```
2. **Parse FASTA files**  
   ```
   python parse_fasta_for_run_inference.py --bin_bedtools=path_to_bedtools2/bin/bedtools \
   --output_file_dir=./dataset/external_dataset/your_folder/ \
   --fasta_path=./dataset/external_dataset/your_folder/ \
   --fasta_query_name=query.fa \
   --fasta_target_name=target.fa \
   --name_analysis=temp
   ```
3. **Download embeddings**  
   ```
   python download_embeddings.py --batch_size=1 \
   --path_to_embedding_query_dir=./dataset/external_dataset/your_folder/temp \
   --embedding_dir=./dataset/external_dataset/your_folder/temp/embeddings
   ```
4. **Activate RIME environment**  
   ```
   conda activate rime
   ```
5. **Run inference**  
   ```
   python run_inference.py --pairs_path=./dataset/external_dataset/your_folder/temp --model_name=RIMEfull
   ```
6. **Parse output**  
   ```
   python parse_output_for_inference.py --inference_dir=./dataset/external_dataset/your_folder/
   ```

### Output:  
After running these steps, you will find the following inside `./dataset/external_dataset/your_folder/`:  
- **`output_table.bedpe`**  
- **`plots/` (a folder containing visualizations)**  

### Note:
- The **25B multi-species NT model** (~10GB) will be **downloaded automatically** the first time you run `download_embeddings.py`.  
- The model weights will be stored in `./NT_dependencies/checkpoints/`.  
