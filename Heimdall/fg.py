from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pandas.api.typing import NAType

from Heimdall.utils import check_states, conditional_print, pca_reduction

if TYPE_CHECKING:
    from Heimdall.tokenizer import TokenizerContext


class Fg(ABC):
    """Abstraction of the gene embedding mapping paradigm.

    Args:
        d_embedding: dimensionality of embedding for each gene entity

    """

    def __init__(
        self,
        context: "TokenizerContext",
        embedding_parameters: DictConfig,
        d_embedding: int,
        vocab_size: int,
        pad_value: int = None,
        mask_value: int = None,
        frozen: bool = False,
        rng: int | np.random.Generator = 0,
        do_pca_reduction: bool = True,
    ):
        self.context = context
        self.d_embedding = d_embedding
        self.embedding_parameters = OmegaConf.to_container(embedding_parameters, resolve=True)
        self.vocab_size = vocab_size
        self.pad_value = vocab_size - 2 if pad_value is None else pad_value
        self.mask_value = vocab_size - 1 if mask_value is None else mask_value
        self.frozen = frozen
        self.rng = np.random.default_rng(rng)
        self.do_pca_reduction = do_pca_reduction
        self._identity_embedding_index = None
        self._identity_valid_mask = None

    @abstractmethod
    @check_states(adata=True)
    def preprocess_embeddings(self, float_dtype: str = "float32"):
        """Preprocess gene embeddings and store them for use during model
        inference.

        Preprocessing may include anything from downloading gene embeddings from
        a URL to generating embeddings from scratch.

        Args:
            float_dtype: dtype to be used for identity embedding state.

        Returns:
            Sets `self.gene_embeddings`.
            Sets tokenizer-local identity state keyed by `context.raw_gene_names`:
            `identity_embedding_index` and `identity_valid_mask`.

        """

    def __getitem__(self, gene_names: Sequence[str], return_mask: bool = False) -> Sequence[int | NAType]:
        """Get the indices of genes in the embedding array.

        Must run `self.preprocess_embeddings()` before using this function.

        Args:
            gene_names: name of the gene as stored in `self.adata`.

        Returns:
            Index of gene in the embedding, or `pd.NA` if the gene has no mapping.

        """
        if self._identity_embedding_index is None or self._identity_valid_mask is None:
            raise ValueError("`Fg` must be preprocessed before use.")

        embedding_indices = self._identity_embedding_index.loc[gene_names]
        valid_mask = self._identity_valid_mask.loc[gene_names]
        if (valid_mask.sum() != len(gene_names)) and not return_mask:
            raise KeyError(
                "At least one gene is not mapped in this `Fg`. "
                "Please remove such genes from consideration in the `Fc`.",
            )

        if return_mask:
            return embedding_indices, valid_mask
        else:
            return embedding_indices

    @property
    def identity_valid_mask(self):
        if self._identity_valid_mask is None:
            return None
        return self._identity_valid_mask.to_numpy()

    @property
    def identity_embedding_index(self):
        if self._identity_embedding_index is None:
            return None
        return self._identity_embedding_index

    @identity_valid_mask.setter
    def identity_valid_mask(self, val):
        val = np.asarray(val, dtype=bool)
        self.vocab_size -= len(self.context.raw_gene_names) - np.sum(val)
        self._identity_valid_mask = pd.Series(val, index=self.context.raw_gene_names, dtype=bool)

    @identity_embedding_index.setter
    def identity_embedding_index(self, val):
        self._identity_embedding_index = pd.Series(val, index=self.context.raw_gene_names)

    def prepare_embedding_parameters(self):
        """Replace config placeholders with values after preprocessing."""
        args = self.embedding_parameters.get("args", {})

        for key, value in args.items():
            if value == "vocab_size":
                value = self.vocab_size  # <PAD> and <MASK> TODO: data.vocab_size
            elif value == "gene_embeddings":
                gene_embeddings = torch.tensor(self.gene_embeddings)  # TODO: type is inherited from NDArray
                pad_vector = torch.zeros(1, self.d_embedding)
                mask_vector = torch.zeros(1, self.d_embedding)
                value = torch.cat((gene_embeddings, pad_vector, mask_vector), dim=0)
                self.pad_value = value.shape[0] - 2
                self.mask_value = value.shape[0] - 1
            else:
                continue

            self.embedding_parameters["args"][key] = value

    def load_from_cache(
        self,
        identity_embedding_index: NDArray,
        identity_valid_mask: NDArray,
        gene_embeddings: NDArray | None,
    ):
        """Load processed values from cache."""
        # TODO: add tests

        self.identity_embedding_index = identity_embedding_index
        self.identity_valid_mask = identity_valid_mask
        self.gene_embeddings = gene_embeddings

        self.prepare_embedding_parameters()

    @property
    def adata(self):
        return self.context.adata


class PretrainedFg(Fg, ABC):
    """Abstraction for pretrained `Fg`s that can be loaded from disk.

    Args:
        embedding_filepath: filepath from which to load pretrained embeddings

    Raises:
        ValueError: if `config.d_embedding` is larger than embedding dimensionality given in filepath.

    """

    def __init__(
        self,
        context: "TokenizerContext",
        # adata: ad.AnnData,
        embedding_parameters: OmegaConf,
        embedding_filepath: Optional[str | PathLike] = None,
        **fg_kwargs,
    ):
        super().__init__(context, embedding_parameters=embedding_parameters, **fg_kwargs)
        self.embedding_filepath = Path(embedding_filepath)

    @abstractmethod
    def load_embeddings(self) -> Dict[str, NDArray]:
        """Load the embeddings from disk and process into map.

        Returns:
            A mapping from gene names to embedding vectors.

        """

    @check_states(adata=True)
    def preprocess_embeddings(self, float_dtype: str = "float32"):
        embedding_map = self.load_embeddings()

        first_embedding = next(iter(embedding_map.values()))
        if len(first_embedding) < self.d_embedding:
            raise ValueError(
                f"Dimensionality of pretrained embeddings ({len(first_embedding)} is less than the embedding "
                "dimensionality specified in the config ({self.d_embedding}). Please decrease the embedding"
                "dimensionality to be compatible with the pretrained embeddings.",
            )

        if len(first_embedding) > self.d_embedding:
            conditional_print(
                f"> Warning, the `Fg` embedding dim {first_embedding.shape} is larger than the model "
                f"dim {self.d_embedding}, truncation may occur.",
                condition=self.context.verbose,
            )

            if self.do_pca_reduction:
                original_embedding_filepath = self.embedding_filepath
                self.embedding_filepath = (
                    original_embedding_filepath.parent
                    / f"{original_embedding_filepath.stem}_reduced_{self.d_embedding}.pt"
                )
                if self.embedding_filepath.is_file():
                    embedding_map = self.load_embeddings()
                    conditional_print(
                        "> Loaded PCA-reduced `Fg` embeddings from cache.",
                        condition=self.context.verbose,
                    )
                else:
                    embedding_map = pca_reduction(embedding_map, n_components=self.d_embedding)
                    torch.save(
                        {gene_name: torch.from_numpy(embedding) for gene_name, embedding in embedding_map.items()},
                        self.embedding_filepath,
                    )
                    conditional_print(
                        "> Used PCA to reduce `Fg` embeddings and cached for future use.",
                        condition=self.context.verbose,
                    )

                self.embedding_filepath = original_embedding_filepath

        valid_gene_names = list(embedding_map.keys())
        source_gene_names = pd.Index(self.context.raw_gene_names)

        valid_mask = pd.array(source_gene_names.isin(valid_gene_names))
        num_mapped_genes = valid_mask.sum()
        (valid_indices,) = np.nonzero(valid_mask)

        index_map = valid_mask.astype(pd.Int64Dtype())
        index_map[~valid_mask] = None
        index_map[valid_indices] = np.arange(num_mapped_genes)

        self.identity_embedding_index = index_map
        self.identity_valid_mask = valid_mask.to_numpy()

        self.gene_embeddings = np.zeros((num_mapped_genes, self.d_embedding), dtype=float_dtype)

        for gene_name in source_gene_names:
            embedding_index = self.identity_embedding_index.loc[gene_name]
            if not pd.isna(embedding_index):
                self.gene_embeddings[embedding_index] = embedding_map[gene_name][: self.d_embedding]

        self.prepare_embedding_parameters()

        conditional_print(
            f"Found {len(valid_indices)} genes with mappings out of {len(source_gene_names)} genes.",
            condition=self.context.verbose,
        )

        map_ratio = len(valid_indices) / len(source_gene_names)
        if map_ratio < 0.5:
            raise ValueError(
                "Very few genes in the dataset are mapped by the `Fg`."
                "Please check if the species is set correctly in the config.",
            )


class IdentityFg(Fg):
    """Identity mapping of gene names to embeddings.

    This is the simplest possible Fg; it implies the use of learnable gene
    embeddings that are initialized randomly, as opposed to the use of
    pretrained embeddings.

    """

    @check_states(adata=True)
    def preprocess_embeddings(self, float_dtype: str = "float32"):
        self.gene_embeddings = None
        self.identity_embedding_index = np.arange(len(self.context.raw_gene_names))
        self.identity_valid_mask = np.full(len(self.context.raw_gene_names), True)

        self.prepare_embedding_parameters()


class TorchTensorFg(PretrainedFg):
    """Mapping of gene names to pretrained embeddings stored as PyTorch
    tensors."""

    def load_embeddings(self):
        raw_gene_embedding_map = torch.load(self.embedding_filepath, weights_only=True)

        raw_gene_embedding_map = {
            gene_name: embedding.detach().cpu().numpy() for gene_name, embedding in raw_gene_embedding_map.items()
        }

        return raw_gene_embedding_map


class CSVFg(PretrainedFg):
    """Mapping of gene names to pretrained Gene2Vec embeddings."""

    def load_embeddings(self):
        raw_gene_embedding_dataframe = pd.read_csv(self.embedding_filepath, sep=r"\s+", header=None, index_col=0)
        raw_gene_embedding_map = {
            gene_name: raw_gene_embedding_dataframe.loc[gene_name].values
            for gene_name in raw_gene_embedding_dataframe.index
        }

        return raw_gene_embedding_map


class Gene2VecFg(TorchTensorFg):
    """Mapping of gene names to pretrained Gene2VecFg embeddings."""


class GenePTFg(TorchTensorFg):
    """Mapping of gene names to pretrained GenePT embeddings."""


class ESM2Fg(TorchTensorFg):
    """Mapping of gene names to pretrained ESM2 embeddings."""


class HyenaDNAFg(TorchTensorFg):
    """Mapping of gene names to pretrained HyenaDNA embeddings."""
