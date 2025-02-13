# RNA-RNA interactions classifier
This repo bla bla

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


## 5. Inference
Put these files inside the directory dataset/external_dataset/your_folder/

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

Run these scripts from the src directory in the following order:
- conda activate download_embeddings 
- cd your_path/src
- nohup python parse_fasta_for_run_inference.py --output_file_dir=dataset/external_dataset/your_folder/--fasta_path=dataset/external_dataset/your_folder/ --fasta_query_name=query.fa --fasta_target_name=target.fa --name_analysis=temp &> parse_fasta_for_run_inference.out &
- nohup python download_embeddings.py --batch_size=1 --path_to_embedding_query_dir=dataset/external_dataset/your_folder/temp --embedding_dir=dataset/external_dataset/your_folder/temp/embeddings &> download_embeddings.out &
- conda activate rime
- nohup python run_inference_new.py --pairs_path=dataset/external_dataset/your_folder/temp --model_name=RIMEfull &> run_inference_new.out &
- nohup python parse_output_for_inference.py --inference_dir=dataset/external_dataset/your_folder/ &> parse_output_for_inference.out &

You will have output_table.bedpe, plots folder inside the dataset/external_dataset/your_folder/ path


# TODO
- risolvi not-found-error in download_embeddings.py
- Link per scaricare NT? oppure e lo stesso allocato da loro? Ho lanciato il download_embeddings.py script e ha scaricato il modello nel folder corretto, verifica sia quello giusto, cioe che i risultati siano uguali a external_dataset/pulldown/try/. Nel caso, possiamo non dare indicazioni, tanto si scarica in automatico, ma conviene scriverlo nel readme per gli utenti che cio avviene.
- metti indicazione su bedtools
- Fai una prova 
- Serve 2. Data?
- elimina script e folder inutili da /dataset/
- elimina script e folder inutili da /models/
- elimina script e folder inutili da /util/
- Metti un link per scaricare il mio modello 
- Crea comandi per rime environment (quello del mio modello che fa le predizioni), tenendo il minimo indispensabile