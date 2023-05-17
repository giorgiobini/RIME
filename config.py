import os

MAX_RNA_SIZE = 5970 #other accepted sequence length (without N) are the ones <= 5974, and also 5976, 5977, 5978, 5979, 5982, 5983, 5984, 5988, 5989, 5994
EMBEDDING_DIM = 2560 #nucleotide-transformer embedding (number of features for each token)

ROOT_DIR =  '/data01/giorgio/RNARNA-NT/' #/data01/gbini/projects/RNA-RNA/
original_files_dir = os.path.join(ROOT_DIR, 'dataset', 'original_files')
rna_rna_files_dir = os.path.join(ROOT_DIR, "dataset", "rna_rna_pairs")
processed_files_dir = os.path.join(ROOT_DIR, "dataset", "processed_files")
nt_data_dir = os.path.join(processed_files_dir, "nt_data")
embedding_dir = os.path.join(nt_data_dir, "embeddings")
metadata_dir = os.path.join(nt_data_dir, "metadata")
nt_dir =  os.path.join(ROOT_DIR, "NT_dependencies")