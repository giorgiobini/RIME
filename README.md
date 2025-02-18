# RNA-RNA interactions classifier
This repository provides instructions, including dependencies and scripts, for using RIME (RNA Interactions Model with Embeddings), a deep learning framework designed for predicting RNA-RNA interactions.

<img src="RIMElogo.jpg">

If this source code is helpful for your research please cite the following publication:

"Decoding RNA-RNA Interactions: The Role of Low-Complexity Repeats and a Deep Learning Framework for Sequence-Based Prediction"

Adriano Setti*†, Giorgio Bini*†, Valentino Maiorca, Flaminia Pellegrini, Gabriele Proietti, Dimitrios Miltiadis-Vrachnos, Alexandros Armaos, Julie Martone, Michele Monti, Giancarlo Ruocco, Emanuele Rodolà, Irene Bozzoni, Alessio Colantoni‡, Gian Gaetano Tartaglia‡

*† Co-first authors
‡ Co-last authors

https://www.biorxiv.org/content/10.1101/2025.02.16.638500v1

## 1. Environment setup 
We recommend you to build a python virtual environment with Anaconda.
The machine has cuda version 525, toolkit cuda 12.0, cudnn 8.8.1.3

#### 1.1 Install miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
export PATH="/home/username/miniconda/bin:$PATH"
```

#### 1.2 Create and activate a new virtual environment, Install the package and other requirements


(Optional)
```
conda install -c conda-forge jupyterlab -y
```


NUCLEOTIDE TRANSFORMER (Required)

```
conda create -n rnarna python=3.10 ipython 
conda activate rnarna
conda install pandas=1.5.3 -y
conda install pytorch torchvision -c pytorch -y
conda install -c anaconda scikit-learn=1.2.2 -y
conda install -c anaconda seaborn=0.12.2 -y
conda install -c conda-forge matplotlib=3.7.1 -y
conda install -c conda-forge matplotlib-venn -y
conda install -c conda-forge tqdm=4.65.0 -y
conda install -c conda-forge ipywidgets=8.0.4 -y
conda install -c conda-forge biopython=1.81 -y
pip install StrEnum==0.4.8
cd NT_dependencies
git clone https://github.com/instadeepai/nucleotide-transformer.git
mv nucleotide-transformer nt
mv nt/* .
rm -r nt
cp mypretrained.py nucleotide_transformer/.
pip install .
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install -c conda-forge pytorch-gpu=2.0.0 -y (C e una ripetizione del secondo comando; jax dovrebbe comunque funzionare anche dopo questo comando, prova a metterlo prima la prossima volta evitando la ripetizione)
pip install datasets


OPPURE: 
copia incolla il folder rnarna dentro la cartella qualcosa/ENTER/envs/
``` 

RIME (Required)

```
conda create -n rime --file spec-file-dnabert.txt
conda activate rime
cd DNABERT_dependencies
python3 -m pip install --editable .
cd ..
pip install -r requirements-dnabert.txt
conda install munch
conda install -c conda-forge biopython=1.79 -y
pip install datasets
conda install -c conda-forge matplotlib-venn -y (I still have todo this in bluecheer)
python -m pip install --no-cache-dir ortools -y  (I still have todo this in bluecheer)

OPPURE: 
copia incolla il folder dnabert dentro la cartella qualcosa/ENTER/envs/
cd DNABERT_dependencies
python3 -m pip install --editable .
```

#### 1.3 Install bedtools2
The inference scripts require bedtools2 installed on your machine.

## 2. Model
We need the data to be structured as the example below.

```
checkpoints
│ 
└── RIMEfull
```

You can download the RIMEfull folder from this link (put a link).


## 3. Inference
Put these files inside the directory your_path/dataset/external_dataset/your_folder/

```
query.fa

>ENSFAKE001
CGUUUCGCUAAACUCUG
>ENSFAKE002
UCGCGAGGCGCAACGGCGCCGACCGAGUGUAGGC


target.fa

>ENSFAKE003
GUGAACGUCGCGAUAGGCGGAACAA
>ENSFAKE004
AGUAACAACGCUAGGUGCGAGUGUCGUC
```

Run these scripts from the your_path/src directory in the following order:
- conda activate download_embeddings 
- python parse_fasta_for_run_inference.py --bin_bedtools=path_to_bedtools2/bin/bedtools --output_file_dir=your_path/dataset/external_dataset/your_folder/--fasta_path=your_path/dataset/external_dataset/your_folder/ --fasta_query_name=query.fa --fasta_target_name=target.fa --name_analysis=temp
- python download_embeddings.py --batch_size=1 --path_to_embedding_query_dir=dataset/external_dataset/your_folder/temp --embedding_dir=dataset/external_dataset/your_folder/temp/embeddings
- conda activate rime
- python run_inference.py --pairs_path=your_path/dataset/external_dataset/your_folder/temp --model_name=RIMEfull
- python parse_output_for_inference.py --inference_dir=your_path/dataset/external_dataset/your_folder/

You will have output_table.bedpe and plots folder inside the your_path/dataset/external_dataset/your_folder/ path

## Dataset 
You can download the dataset for training your own model from this link (put a link).
You can download the Test Set (200x200) for reproducibility from this link (put a link).

# TODO
- metti gradcam?
- Fai un test ora che ho pulito il file config.py
- Metti un link per scaricare il mio modello 
- Crea comandi per rime environment (quello del mio modello che fa le predizioni), tenendo il minimo indispensabile