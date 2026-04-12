from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from scipy.sparse import issparse

from Heimdall.fe import Fe
from Heimdall.fg import Fg
from Heimdall.utils import instantiate_from_config

if TYPE_CHECKING:
    from Heimdall.cell_representations import CellRepresentation


class Fc:
    """Abstraction for cell embedding.

    Args:
        fg: `Fg` used for this `Fc` implementation.
        fe: `Fe` used for this `Fc` implementation.
        max_input_length: maximum number of identity/expression tokens to consider for each cell.
            Extra tokens are limited.

    """

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        data: "CellRepresentation",
        tailor_config: DictConfig,
        order_config: DictConfig,
        reduce_config: DictConfig,
        embedding_parameters: DictConfig,
        max_input_length: Optional[int] = None,
        float_dtype: str = "float32",
        rng: int | np.random.Generator = 0,
    ):
        self.fg = fg
        self.fe = fe
        self.data = data
        self.max_input_length = max_input_length
        self.float_dtype = float_dtype
        self.embedding_parameters = OmegaConf.to_container(embedding_parameters, resolve=True)
        self.rng = np.random.default_rng(rng)
        self.extra_keys = set()

        self.tailor = instantiate_from_config(tailor_config, fc=self)
        self.order = instantiate_from_config(order_config, fc=self)
        self.reduce = instantiate_from_config(reduce_config, fc=self)

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Retrieve `identity_inputs`, `expression_inputs` and
        `expression_padding`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """

        if cell_index == -1:  # Dummy `cell_index`
            identity_inputs = pd.array(np.full(self.max_input_length, self.fg.pad_value), dtype="Int64")
            expression_inputs = np.full(self.max_input_length, self.fe.pad_value)
        else:
            identity_indices, expression_inputs = self.fe[cell_index]

            gene_list = self.data.gene_names[identity_indices]  # convert to ENSEMBL Gene Names
            identity_inputs = self.fg[gene_list]  # convert the genes into fg

            if len(identity_inputs) != len(expression_inputs):
                raise ValueError(
                    "Gene identity and expression inputs do not have the same shape; `Fg` and `Fe` are incompatible.",
                )

            # first, drop any `NaN` values here
            # Assuming gene_tokenization is a pandas IntegerArray and expression_tokenization is a numpy array
            # TODO: what does `NaN` represent here?
            valid_mask = ~np.isnan(expression_inputs)

            identity_inputs = identity_inputs[valid_mask].to_numpy()
            # identity_indices = identity_indices[valid_mask]
            expression_inputs = expression_inputs[valid_mask]

            # gene_order = self.order(cell_index, identity_indices, expression_inputs)
            gene_order = self.order(identity_inputs, expression_inputs)

            # Padding and truncating
            identity_inputs, expression_inputs = self.tailor(
                identity_inputs,
                expression_inputs,
                gene_order,
            )

        expression_padding = expression_inputs == self.fe.pad_value

        outputs = {
            "identity_inputs": identity_inputs,
            "expression_inputs": expression_inputs,
            "expression_padding": expression_padding,
        }

        return outputs

    @property
    def adata(self):
        return self.data.adata


class ChromosomeAwareFc(Fc):
    def __init__(
        self,
        *fc_args,
        gene_metadata_filepath: str | Path,
        ensembl_dir: str | Path,
        species: str,
        **fc_kwargs,
    ):
        """
        Args:
            gene_metadata_filepath: path to gene metadata .csv
            ensembl_dir: path to directory in which Ensembl mapping file is stored
            species: species from which single-cell dataset is derived
        """

        super().__init__(*fc_args, **fc_kwargs)

        self.gene_metadata = pd.read_csv(gene_metadata_filepath)
        self.ensembl_dir = ensembl_dir
        self.species = species

        self.gene_metadata["spec_chrom"] = pd.Categorical(
            self.gene_metadata["species"] + "_" + self.gene_metadata["chromosome"],
        )

        # https://github.com/snap-stanford/UCE/blob/8227a65cdd021b9186ef86671d2aef5c895c8e4b/data_proc/data_utils.py#L155
        # TODO: load chromosome one-hot encoding and start positions for all genes

        # symbol_to_ensembl_mapping = symbol_to_ensembl_from_ensembl(
        #     data_dir=self.ensembl_dir,
        #     genes=spec_chrom.index.tolist(),
        #     species=self.species,
        # )
        # spec_chrom.index = spec_chrom.index.map(symbol_to_ensembl_mapping.mapping_reduced)
        self.extract_gene_positions()
        self.chrom_token_offset = 1

    def extract_gene_positions(self):
        spec_chrom = self.gene_metadata[self.gene_metadata["species"] == self.species].set_index("gene_symbol")
        try:
            # NOTE: below is different from UCE...
            gene_names = [k.upper() for k in self.adata.var["gene_symbol"]]
            # gene_chrom = spec_chrom.loc[gene_names]
            gene_chrom = spec_chrom.reindex(gene_names, copy=True)
        except KeyError as e:
            raise ValueError(
                "Input AnnData cannot contain gene names that are unmapped in the chromosome metadata.",
            ) from e

        # TODO: for pretraining, we should keep extraneous codes (i.e. no `remove_unused_categories()`)
        dataset_chroms = gene_chrom["spec_chrom"].cat.remove_unused_categories().cat.codes
        # print("Max Code:", max(dataset_chroms))
        dataset_pos = gene_chrom["start"].values

        self.unique_chromosomes = np.unique(dataset_chroms)

        self.chroms = dataset_chroms
        self.starts = dataset_pos

    # @Fc.adata.setter
    # def adata(self, val):
    #     Fc.adata.fset(self, val)
    #     self.extract_gene_positions()


class DummyFc(Fc):
    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        data: "CellRepresentation",
        # adata: ad.AnnData,
        tailor_config: DictConfig,
        order_config: DictConfig,
        reduce_config: DictConfig,
        embedding_parameters: DictConfig,
        max_input_length: Optional[int] = None,
        float_dtype: str = "float32",
        rng: int | np.random.Generator = 0,
    ):
        self.fg = fg
        self.fe = fe
        # self.adata = adata
        self.max_input_length = max_input_length
        self.extra_keys = set()

    """Dummy `Fc` that does not tailor the size of the input."""

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Dummy `__getitem__` for model that does not need an `Fc`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """
        identity_indices, expression_inputs = self.fe[cell_index]
        expression_padding = np.zeros(self.max_input_length)

        outputs = {
            "identity_indices": identity_indices,
            "expression_inputs": expression_inputs,
            "expression_padding": expression_padding,
        }

        return outputs


class NicheformerFc(Fc):
    """FC variant that emits full gene-space expression and metadata codes.

    This mirrors the metadata/data-shaping behavior required by the Nicheformer
    encoder while keeping the SCFM path inside Heimdall.

    """

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        data: "CellRepresentation",
        tailor_config: DictConfig,
        order_config: DictConfig,
        reduce_config: DictConfig,
        embedding_parameters: DictConfig,
        max_input_length: Optional[int] = None,
        float_dtype: str = "float32",
        rng: int | np.random.Generator = 0,
    ):
        self.fg = fg
        self.fe = fe
        self.data = data
        self.max_input_length = max_input_length
        self.float_dtype = float_dtype
        self.embedding_parameters = OmegaConf.to_container(embedding_parameters, resolve=True)
        self.rng = np.random.default_rng(rng)
        self.extra_keys = {"technology", "species", "modality", "assay"}
        if not hasattr(self.data, "nicheformer_metadata_codebooks"):
            self.data.nicheformer_metadata_codebooks = {}

    def __getitem__(self, cell_index: int):
        valid_mask = self._get_identity_valid_mask()
        num_genes = int(valid_mask.sum())

        if cell_index == -1:
            outputs = {
                "identity_inputs": np.zeros((1,), dtype=np.int64),
                "expression_inputs": np.zeros((num_genes,), dtype=np.float32),
                "expression_padding": np.ones((1,), dtype=bool),
                "technology": np.array(-1, dtype=np.int64),
                "species": np.array(-1, dtype=np.int64),
                "modality": np.array(-1, dtype=np.int64),
            }
            if "assay" in self.adata.obs:
                outputs["assay"] = np.array(-1, dtype=np.int64)
            return outputs

        expression = self.adata.X[cell_index, valid_mask]
        if issparse(expression):
            expression = expression.toarray().ravel()
        else:
            expression = np.asarray(expression).ravel()

        expression = expression.astype(np.float32, copy=False)
        panel_idx = getattr(self.data.fe, "gene_panel_idx", None)
        if panel_idx is not None:
            panel_idx = np.asarray(panel_idx, dtype=np.int64)
            masked_expression = np.zeros_like(expression)
            masked_expression[panel_idx] = expression[panel_idx]
            expression = masked_expression

        outputs = {
            "identity_inputs": np.zeros((1,), dtype=np.int64),
            "expression_inputs": expression,
            "expression_padding": np.zeros((1,), dtype=bool),
            "technology": self._get_metadata_code("technology", cell_index),
            "species": self._get_metadata_code("species", cell_index),
            "modality": self._get_metadata_code("modality", cell_index),
        }
        if "assay" in self.adata.obs:
            outputs["assay"] = self._get_metadata_code("assay", cell_index)

        return outputs

    def _get_identity_valid_mask(self):
        if self.fg is not None and hasattr(self.fg, "identity_valid_mask"):
            return np.asarray(self.fg.identity_valid_mask)

        return np.ones(self.adata.n_vars, dtype=bool)

    def _get_metadata_code(self, field: str, cell_index: int) -> np.int64:
        if field not in self.adata.obs:
            raise ValueError(f"`adata.obs[{field!r}]` is required for NicheformerFc.")

        series = self.adata.obs[field].astype("category")
        categories = [str(category) for category in series.cat.categories]
        if not hasattr(self.data, "nicheformer_metadata_codebooks"):
            self.data.nicheformer_metadata_codebooks = {}
        if field not in self.data.nicheformer_metadata_codebooks:
            self.data.nicheformer_metadata_codebooks[field] = categories

        return np.int64(series.cat.codes.iloc[int(cell_index)])
