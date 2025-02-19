import os

MAX_RNA_SIZE = 5970 

MODEL_NAME = 'RIME'

EMBEDDING_DIM = 2560 #nucleotide-transformer embedding (number of features for each token)

MIN_RNA_SIZE_DATALOADER = 100

ROOT_DIR =  os.path.dirname(os.path.abspath(__file__))
dataset_files_dir = os.path.join(ROOT_DIR, 'dataset')

#NT
nt_dir =  os.path.join(ROOT_DIR, "NT_dependencies")