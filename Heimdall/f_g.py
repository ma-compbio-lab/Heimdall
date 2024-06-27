## loading in libraries
import scanpy as sc
import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import hydra
import pandas as pd
from omegaconf import OmegaConf


#####
# an example of some custom fg/fcs
#####
def identity_fg(adata_var):
    """
    identity_fg is an fg that returns a token id for each gene, effectively each gene
    is its own word.

    args:
        - adata_var: takes in the var dataframe, in this case, it expects the index to have the gene names

    output:
        - the output is a dictionary map between the gene names, and their corersponding token id for nn.embedding
    """
    print("> Performing the f_g identity, desc: each gene is its own token")
    gene_df = adata_var
    gene_mapping = {label: idx for idx, label in enumerate(gene_df.index.unique(), start=0)}
    return gene_mapping
