import os

MAX_RNA_SIZE = 5989
EMBEDDING_DIM = 2560 #nucleotide-transformer embedding (number of features for each token)

ROOT_DIR =  '/data01/gbini/projects/RNA-RNA/' #'/data01/giorgio/RNARNA-NT/' #os.path.dirname(os.path.abspath('.'))
original_files_dir = os.path.join(ROOT_DIR, 'dataset', 'original_files')
rna_rna_files_dir = os.path.join(ROOT_DIR, "dataset", "rna_rna_pairs")
processed_files_dir = os.path.join(ROOT_DIR, "dataset", "processed_files")
nt_data_dir = os.path.join(processed_files_dir, "nt_data")
embedding_dir = os.path.join(nt_data_dir, "embeddings")
metadata_dir = os.path.join(nt_data_dir, "metadata")
nt_dir =  os.path.join(ROOT_DIR, "NT_dependencies")