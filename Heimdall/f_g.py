from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas.api.typing import NAType


class Fg(ABC):
    """Abstraction of the gene embedding mapping paradigm.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.

    """

    def __init__(self, adata: ad.AnnData, config: dict):
        self.adata = adata
        _, self.num_genes = adata.shape
        self.config = config

    @abstractmethod
    def preprocess_embeddings(self):
        """Preprocess gene embeddings and store them for use during model
        inference.

        Preprocessing may include anything from downloading gene embeddings from
        a URL to generating embeddings from scratch.

        """

    def __getitem__(self, gene_names: Sequence[str]) -> int | NAType:
        """Get the indices of genes in the embedding array.

        Args:
            gene_names: name of the gene as stored in `self.adata`.

        Returns:
            Index of gene in the embedding, or `pd.NA` if the gene has no mapping.

        """
        embedding_indices = self.adata.var.loc[gene_names, "embedding_index"]
        if np.any(embedding_indices.isna()):
            raise KeyError(
                f"At least one gene is not mapped in this Fg. Please remove such genes from "
                "consideration in the Fc.",
            )

        return embedding_indices


class PretrainedFg(Fg, ABC):
    """Abstraction for pretrained `Fg`s that can be loaded from disk."""

    @abstractmethod
    def load_embeddings(self) -> Dict[str, NDArray]:
        """Load the embeddings from disk and process into map.

        Returns:
            A mapping from gene names to embedding vectors.

        """

    def preprocess_embeddings(self):
        embedding_map = self.load_embeddings()
        valid_gene_names = list(embedding_map.keys())

        print(self.adata.var_names[:10])
        print(valid_gene_names[:10])
        valid_mask = pd.array(np.isin(self.adata.var_names.values, valid_gene_names))
        num_mapped_genes = valid_mask.sum()
        print(num_mapped_genes)
        (valid_indices,) = np.nonzero(valid_mask)

        index_map = valid_mask.astype(pd.Int64Dtype())
        index_map[~valid_mask] = None
        index_map[valid_indices] = np.arange(num_mapped_genes)

        self.adata.var["embedding_index"] = index_map
        self.adata.var["valid_mask"] = valid_mask

        self.gene_embeddings = np.zeros((num_mapped_genes, self.config.d_embedding), dtype=np.float64)

        for gene_name in self.adata.var_names:
            embedding_index = self.adata.var.loc[gene_name, "embedding_index"]
            if not pd.isna(embedding_index):
                self.gene_embeddings[embedding_index] = embedding_map[gene_name][: self.config.d_embedding]

        print(f"Found {len(valid_indices)} genes with mappings out of {len(self.adata.var_names)} genes.")


class IdentityFg(Fg):
    """Identity mapping of gene names to embeddings.

    This is the simplest possible Fg; it implies the use of learnable gene
    embeddings that are initialized randomly, as opposed to the use of
    pretrained embeddings.

    """

    def preprocess_embeddings(self):
        self.gene_embeddings = None
        self.adata.var["embedding_index"] = np.arange(self.num_genes)
        self.adata.var["valid_mask"] = np.full(self.num_genes, True)


class ESM2Fg(PretrainedFg):
    """Mapping of gene names to pretrained ESM2 embeddings."""

    def load_embeddings(self):
        raw_gene_embedding_map = torch.load(self.config.embedding_filepath)

        raw_gene_embedding_map = {
            gene_name: embedding.detach().cpu().numpy() for gene_name, embedding in raw_gene_embedding_map.items()
        }

        return raw_gene_embedding_map


class Gene2VecFg(PretrainedFg):
    """Mapping of gene names to pretrained Gene2Vec embeddings."""

    def load_embeddings(self):
        raw_gene_embedding_dataframe = pd.read_csv(self.config.embedding_filepath, sep=r"\s+", header=None, index_col=0)
        raw_gene_embedding_map = {
            gene_name: raw_gene_embedding_dataframe.loc[gene_name].values
            for gene_name in raw_gene_embedding_dataframe.index
        }

        return raw_gene_embedding_map


def identity_fg(adata_var, species="human"):
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


def gene2vec_fg(adata_var, species="human"):
    if species != "human":
        raise ValueError(f"Unsupported species: {species}. This function only supports 'human'.")

    file_path = "/work/magroup/shared/Heimdall/data/pretrained_embeddings/gene2vec/gene2vec_genes.txt"

    # gene_df = adata_var
    gene_to_index = identity_fg(adata_var, "human")
    embedding_list = []

    with open(file_path) as file:
        # skip first line
        # metadata = file.readline().strip()

        # read rest of file line by line
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


EMBEDDING_DIR = Path("/work/magroup/shared/Heimdall/data/pretrained_embeddings/ESM2")

# SPECIES_TO_GENE_EMBEDDING_PATH = {
#          'human': EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
#         'mouse': EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
#     }
SPECIES_TO_GENE_EMBEDDING_PATH = {
    "human": EMBEDDING_DIR / "protein_map_human_ensembl.pt",
    "mouse": EMBEDDING_DIR / "protein_map_mouse_ensembl.pt",
}


def esm2_fg(adata_var, species="human"):
    """esm2_fg is an fg that returns an ESM2 protein embedding for each gene.

    args:
        - adata: takes in the var dataframe, in this case, it expects the index to have the gene names

    output:
        - the output is a dictionary map between the gene names, and their corersponding token id for nn.embedding

    """
    embedding_path = SPECIES_TO_GENE_EMBEDDING_PATH[species]

    protein_gene_map = torch.load(embedding_path)
    gene_to_index = {}
    embedding_list = []

    # Filter adata_var to include only genes with embeddings in protein_gene_map
    valid_genes = [gene for gene in adata_var.index if gene in protein_gene_map]

    print(f"Found {len(valid_genes)} genes with mappings out of {len(adata_var.index)} genes.")

    # Map genes to indices and collect their embeddings
    for idx, gene in enumerate(valid_genes):
        gene_to_index[gene] = idx
        embedding_list.append(protein_gene_map[gene].numpy())

    # Stack embeddings into a matrix
    embedding_matrix = np.vstack(embedding_list)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    # Create the embedding layer from pre-trained embeddings
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix)

    return embedding_layer, gene_to_index
