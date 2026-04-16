from typing import TYPE_CHECKING

import numpy as np
from omegaconf import DictConfig

from Heimdall.utils import instantiate_from_config, issparse

if TYPE_CHECKING:
    from Heimdall.fc import Fc
    from Heimdall.fe import Fe
    from Heimdall.fg import Fg


class TokenizerContext:
    def __init__(self, adata, raw_gene_names, float_dtype: str = "float32", verbose: int = 0):
        self.adata = adata
        self.raw_gene_names = np.asarray(raw_gene_names)
        self.float_dtype = float_dtype
        self.verbose = verbose
        self.fg = None
        self.fe = None
        self.fc = None

    @property
    def identity_valid_mask(self):
        if self.fg.identity_valid_mask is not None:
            return np.asarray(self.fg.identity_valid_mask, dtype=bool)
        return np.ones(len(self.raw_gene_names), dtype=bool)

    @property
    def gene_names(self):
        return self.raw_gene_names[self.identity_valid_mask]

    @property
    def feature_matrix(self):
        return self.adata.X

    @property
    def num_genes(self):
        return len(self.gene_names)

    def set_representation_functions(self, fg=None, fe=None, fc=None):
        if fg is not None:
            self.fg = fg
        if fe is not None:
            self.fe = fe
        if fc is not None:
            self.fc = fc


class KnnTokenizerContext(TokenizerContext):
    """Tokenizer context for precomputed KNN pseudo-expression features.

    Expects the configured `feature_obsm_key` to point to the active tokenizer
    feature matrix in `adata.obsm`. For standard KNN-preprocessed files this is
    a sparse CSR-like matrix. The ordered feature names must come from either:

    - `adata.uns["atlas_knn"]["feature_gene_names"]`
    - DataFrame columns on `adata.obsm[feature_obsm_key]`

    The sparse contract matters for `Fe` paths with `drop_zeros=True`, which
    route through the CSR extraction helper rather than a dense fallback.

    """

    def __init__(
        self,
        adata,
        raw_gene_names,
        float_dtype: str = "float32",
        verbose: int = 0,
        feature_obsm_key: str = "X_atlas_knn_mean_expr",
        feature_gene_names_uns_key: str = "atlas_knn",
    ):
        feature_matrix = adata.obsm.get(feature_obsm_key)
        if feature_matrix is None:
            raise ValueError(f"`adata.obsm[{feature_obsm_key!r}]` is required for KnnTokenizerContext.")

        feature_gene_names = None
        feature_metadata = adata.uns.get(feature_gene_names_uns_key)
        if isinstance(feature_metadata, dict):
            feature_gene_names = feature_metadata.get("feature_gene_names")

        if feature_gene_names is None:
            columns = getattr(feature_matrix, "columns", None)
            if columns is not None:
                feature_gene_names = columns

        if feature_gene_names is None:
            raise ValueError(
                "KnnTokenizerContext requires ordered feature gene names from "
                f"`adata.uns[{feature_gene_names_uns_key!r}]['feature_gene_names']` or DataFrame columns.",
            )

        super().__init__(
            adata=adata,
            raw_gene_names=np.asarray(feature_gene_names),
            float_dtype=float_dtype,
            verbose=verbose,
        )
        self.feature_obsm_key = feature_obsm_key
        self.feature_gene_names_uns_key = feature_gene_names_uns_key

    @property
    def feature_matrix(self):
        feature_matrix = self.adata.obsm[self.feature_obsm_key]
        if issparse(feature_matrix):
            return feature_matrix
        return feature_matrix.to_numpy() if hasattr(feature_matrix, "to_numpy") else np.asarray(feature_matrix)


class Tokenizer:
    def __init__(
        self,
        config: DictConfig,
        adata,
        raw_gene_names,
        rng,
        float_dtype: str = "float32",
        verbose: int = 0,
    ):
        self.config = config
        self.rng = rng
        self.float_dtype = float_dtype
        self.processed = False
        scfm_cfg = config.scfm if "scfm" in config else config
        self.context = instantiate_from_config(
            scfm_cfg.tokenizer_context,
            adata=adata,
            raw_gene_names=raw_gene_names,
            float_dtype=float_dtype,
            verbose=verbose,
        )

        self.fg: Fg
        self.fe: Fe
        self.fc: Fc
        self.fg, _ = instantiate_from_config(
            scfm_cfg.fg,
            self.context,
            vocab_size=len(self.context.raw_gene_names) + 2,
            rng=self.rng,
            return_name=True,
        )
        self.context.set_representation_functions(fg=self.fg)
        self.fe, _ = instantiate_from_config(
            scfm_cfg.fe,
            self.context,
            vocab_size=len(self.context.raw_gene_names) + 2,
            rng=self.rng,
            return_name=True,
        )
        self.fc, _ = instantiate_from_config(
            scfm_cfg.fc,
            self.fg,
            self.fe,
            self.context,
            float_dtype=self.float_dtype,
            rng=self.rng,
            return_name=True,
        )
        self.context.set_representation_functions(fe=self.fe, fc=self.fc)

    def __getitem__(self, cell_index: int):
        return self.fc[cell_index]

    def preprocess_embeddings(self):
        self.fg.preprocess_embeddings()
        self.fe.preprocess_embeddings()
        self.processed = True

    def load_from_cache(
        self,
        identity_embedding_index,
        identity_valid_mask,
        gene_embeddings,
        expression_embeddings,
    ):
        self.fg.load_from_cache(identity_embedding_index, identity_valid_mask, gene_embeddings)
        self.fe.load_from_cache(expression_embeddings)
        self.processed = True

    def get_cache_state(self):
        identity_embedding_index, identity_valid_mask = self.fg.__getitem__(
            self.context.raw_gene_names,
            return_mask=True,
        )
        return (
            identity_embedding_index,
            identity_valid_mask,
            self.fg.gene_embeddings,
            self.fe.expression_embeddings,
        )
