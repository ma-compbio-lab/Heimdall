import numpy as np
import torch 
from pathlib import Path

def identity_fg(adata_var, species='human'):
    """Identify gene function.

    Returns a token id for each gene, effectively each gene is its own word.

    Args:
        adata_var: takes in the var dataframe, in this case, it expects the
            index to have the gene names.

    Return:
        A dictionary map between the gene names, and their corersponding token
        id for nn.embedding.

    """
    print("> Performing the f_g identity, desc: each gene is its own token")
    gene_df = adata_var
    gene_mapping = {label: idx for idx, label in enumerate(gene_df.index.unique(), start=0)}
    return gene_mapping

def gene2vec_fg(adata_var, species='human'):
    if species != 'human':
        raise ValueError(f"Unsupported species: {species}. This function only supports 'human'.")
        
    file_path = '/work/magroup/shared/Heimdall/data/pretrained_embeddings/gene2vec/gene2vec_dim_200_iter_9_w2v.txt' 

    gene_df = adata_var
    gene_to_index = {label: idx for idx, label in enumerate(gene_df['gene_symbol'].unique(), start=0)}
    embedding_list = []

    with open(file_path, 'r') as file:
        #skip first line
        metadata = file.readline().strip()

        #read rest of file line by line
        for line in file:
            parts = line.strip().split()
            gene_name = parts[0]

            if gene_name in gene_to_index:
                embedding = np.array(list(map(float, parts[1:])))
                idx = gene_to_index[gene_name]
                if len(embedding_list) <= idx:
                    embedding_list.extend([None] * (idx + 1 - len(embedding_list)))

                embedding_list[idx] = embedding

    # Identify and remove genes with missing embeddings
    indices_to_remove = [idx for idx, emb in enumerate(embedding_list) if emb is None]
    genes_to_remove = [gene for gene, idx in gene_to_index.items() if idx in indices_to_remove]
    
    for gene in genes_to_remove:
        del gene_to_index[gene]

    # Rebuild the embedding list by removing None entries
    embedding_list = [emb for emb in embedding_list if emb is not None]

    # Optionally, rebuild gene_to_index to ensure contiguous indices
    gene_to_index = {gene: idx for idx, (gene, emb) in enumerate(zip(gene_to_index.keys(), embedding_list))}
    embedding_matrix = np.vstack(embedding_list)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix)
    return embedding_layer, gene_to_index


def esm2_fg(adata_var, species='human'):
    """
    esm2_fg is an fg that returns an ESM2 protein embedding for each gene

    args:
        - adata: takes in the var dataframe, in this case, it expects the index to have the gene names

    output:
        - the output is a dictionary map between the gene names, and their corersponding token id for nn.embedding
    """
    EMBEDDING_DIR = Path('/work/magroup/shared/Heimdall/data/pretrained_embeddings/ESM2')

    SPECIES_TO_GENE_EMBEDDING_PATH = {
             'human': EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
            'mouse': EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
        }
    
    embedding_path = SPECIES_TO_GENE_EMBEDDING_PATH[species]
    
    
    protein_gene_map = torch.load(embedding_path)
    
    gene_intersection =  set(adata_var['gene_symbol']).intersection(protein_gene_map.keys())

    gene_to_index = {gene: idx for idx, gene in enumerate(gene_intersection)}
    embedding_list = [protein_gene_map[gene].numpy() for gene in gene_intersection]

    embedding_matrix = np.vstack(embedding_list)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix)

    return embedding_layer, gene_to_index