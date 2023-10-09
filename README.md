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
``` 

DNABERT (Required)

```
conda create -n dnabert --file spec-file-dnabert.txt
conda activate dnabert
cd DNABERT_dependencies
python3 -m pip install --editable .
cd ..
pip install -r requirements-dnabert.txt
conda install munch
conda install -c conda-forge biopython=1.79 -y
pip install datasets
conda install -c conda-forge matplotlib-venn -y (I still have todo this in bluecheer)
python -m pip install --no-cache-dir ortools -y  (I still have todo this in bluecheer)
```

INTARNA 

``` 
conda create -n intarna 
conda activate intarna
conda install -c conda-forge -c bioconda intarna

create_intarna_fasta.ipynb
cd /data01/giorgio/RNARNA-NT/dataset/processed_files/intarna/test500/
nohup IntaRNA --outCsvCols=id1,start1,end1,id2,start2,end2,subseqDP,hybridDP,E,E_norm -t rna1.fasta -q rna2.fasta --threads=40 --outMode=C --outPairwise -n 5 &> test.csv &

cd /data01/giorgio/RNARNA-NT/dataset/processed_files/intarna/val500/
nohup IntaRNA --outCsvCols=id1,start1,end1,id2,start2,end2,subseqDP,hybridDP,E,E_norm -t rna1.fasta -q rna2.fasta --threads=40 --outMode=C --outPairwise -n 5 &> val.csv &
``` 

nohup IntaRNA --outCsvCols=id1,start1,end1,id2,start2,end2,subseqDP,hybridDP,E,E_norm -t rna1.fasta -q rna2.fasta --threads=5 --outMode=C --outPairwise -n 5 &> mario.csv &


## 2. Data
We need the data to be structured as the example below.

```
dataset
│ 
├── annotation_files
├── processed_files
└── rna_rna_pairs
```

You can download the original_files folder from this link (put a link). If you don't need to train the model you can avoid to download the original_files and keep the directory empty.

## 3. Prepare the data
Data preprocessing steps 

- preprocess_adri_data.ipynb [dnabert]
- train_test_val.ipynb [dnabert]

(OPTIONAL, only needed if you want secondary structure predictions)
- python create_fasta_query_for_secondary_structure.py [dnabert]
- cd ../UFold_dependencies/ [dnabert]
- python run_ufold_predictions.py [dnabert]
- cd ../src/ [dnabert]
- python dot_bracket_preprocessing.py [dnabert]
- Create_df_genes_dot_br.ipynb [dnabert]


## 4. Train your model
Skip this section if you only need inference.

Run these scripts from the src directory in the following order:
- data_augmentation.ipynb (to test if valentino classes work) [rnarna]
- Create_datasets.ipynb (to decide the hyperparameters of the dataloader and create the training, test, validation datasets) [rnarna]
- download_embeddings.py [rnarna]
- nohup python download_embeddings.py --batch_size 19 &> download_embeddings.out & [rnarna]
- train_binary_cl.py [rnarna]
- train_binary_cl2_finetuning.py [rnarna]
- run_binary_cl_on_test2.py OR run_binary_cl_on_test2_500.py [rnarna]
- plot_test2_results.ipynb [rnarna]



## 5. Inference
Put your files inside the directory dataset/external_dataset/your_folder/
You must have these files inside your_folder:

```
embedding_query.csv

id_query,cdna
ENSFAKE001,CGUUUCGCUAAACUCUG
ENSFAKE002,UCGCGAGGCGCAACGGCGCCGACCGAGUGUAGGC
ENSFAKE003,GUGAACGUCGCGAUAGGCGGAACAA
ENSFAKE004,AGUAACAACGCUAGGUGCGAGUGUCGUC



pairs.txt

ENSFAKE001_ENSFAKE002
ENSFAKE001_ENSFAKE003
ENSFAKE003_ENSFAKE004
```

Run these scripts from the src directory in the following order:
- nohup python download_embeddings.py --path_to_embedding_query_dir=/data01/giorgio/RNARNA-NT/dataset/external_dataset/pulldown --embedding_dir=/data01/giorgio/RNARNA-NT/dataset/external_dataset/pulldown/embeddings &> download_embeddings_inference.out &
- run_inference.ipynb