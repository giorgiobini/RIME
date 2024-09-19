import os

MAX_RNA_SIZE = 5970 #other accepted sequence length (without N) are the ones <= 5974, and also 5976, 5977, 5978, 5979, 5982, 5983, 5984, 5988, 5989, 5994

MIN_RNA_SIZE_DATALOADER = 100 # I will not sample with the dataloader rna lengths < MIN_RNA_SIZE_POS_SAMPLES_TRAINING, with the exception of smaller RNAs


MODEL_NAME = 'RRINet'


MAX_RNA_SIZE_BERT = 500

EMBEDDING_DIM_BERT = 768 #bert-transformer embedding (number of features for each token)
EMBEDDING_DIM = 2560 #nucleotide-transformer embedding (number of features for each token)
N_PCA = 1000 #dimension to reduce nucleotide-transformer embeddings

ROOT_DIR =  '/data01/giorgio/RNARNA-NT/' #/data01/gbini/projects/RNA-RNA/
dataset_files_dir = os.path.join(ROOT_DIR, 'dataset')
original_files_dir = os.path.join(dataset_files_dir, 'original_files')
rna_rna_files_dir = os.path.join(dataset_files_dir, "rna_rna_pairs")
processed_files_dir = os.path.join(dataset_files_dir, "processed_files")

#INTARNA
intarna_dir = os.path.join(processed_files_dir, "intarna")

#Logistic regression mapping path
LR_MAPPING_PATH = os.path.join(dataset_files_dir, 'logistic_regression_mapping.pkl')

#NT
nt_data_dir = os.path.join(processed_files_dir, "nt_data")
embedding_dir = os.path.join(nt_data_dir, "embeddings")
metadata_dir = os.path.join(nt_data_dir, "metadata")
nt_dir =  os.path.join(ROOT_DIR, "NT_dependencies")

#UFold
ufold_dir = os.path.join(ROOT_DIR, 'UFold_dependencies')
ufold_path= os.path.join(ufold_dir, 'models', 'ufold_train_alldata.pt')

#DNABERT
bert_pretrained_dir = os.path.join(dataset_files_dir, 'pre_trained_DNABERT', '6-new-12w-0')
dnabert_dir =  os.path.join(ROOT_DIR, "NT_dependencies")