import os
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module

from Heimdall.fc import Fc

# Set HEIMDALL_DEBUG_EMBED_BOUNDS=1 to check embedding indices on the CPU side before they hit
# an nn.Embedding lookup. Out-of-range indices there normally surface as an async, misleadingly-
# located "CUDA error: device-side assert triggered" -- this raises a precise Python exception
# (which array, which value, which table size) at the real point of failure instead. Debug-only;
# safe to delete once the gene_mlm CUDA-assert bug (partition 20M pretraining) is root-caused.
_DEBUG_EMBED_BOUNDS = bool(os.environ.get("HEIMDALL_DEBUG_EMBED_BOUNDS"))


def _find_num_embeddings(layer: Module) -> int | None:
    if hasattr(layer, "num_embeddings"):
        return layer.num_embeddings
    for submodule in layer.modules():
        if hasattr(submodule, "num_embeddings"):
            return submodule.num_embeddings
    return None


def _check_embed_bounds(name: str, indices: Tensor, layer: Module, context: str = ""):
    if not _DEBUG_EMBED_BOUNDS or layer is None:
        return

    num_embeddings = _find_num_embeddings(layer)
    if num_embeddings is None:
        return

    idx = indices.detach()
    idx_long = idx.to(torch.long)
    idx_min = int(idx_long.min().item())
    idx_max = int(idx_long.max().item())

    if idx_min < 0 or idx_max >= num_embeddings:
        bad_mask = (idx_long < 0) | (idx_long >= num_embeddings)
        num_bad = int(bad_mask.sum().item())
        bad_values = idx_long[bad_mask].unique()[:10].tolist()
        raise RuntimeError(
            f"[HEIMDALL_DEBUG_EMBED_BOUNDS] {name!r} indices out of range for its embedding table "
            f"({context}): table has num_embeddings={num_embeddings}, but got min={idx_min}, "
            f"max={idx_max} ({num_bad}/{idx_long.numel()} out-of-range entries, "
            f"sample values={bad_values}). tensor shape={tuple(idx.shape)}, dtype={idx.dtype}.",
        )


class Reduce(ABC):
    def __init__(
        self,
        fc: Fc,
    ):
        self.fc = fc

    @abstractmethod
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cell batch using the embedding layers.

        It can be assumed that both the identity inputs and the expression inputs have been padded/
        limited at this stage, i.e. they are regular-shaped tensors.

        Args:
            identity_inputs: batched gene identity inputs
            gene_embedding_layer: Torch module for embedding based on gene identity.
            expression_inputs: batched gene expression inputs
            expression_embedding_layer: Torch module for embedding based on expression.
            metadata_embedding_layer: Torch module for embedding based on metadata.

        Returns:
            Embeddings of cells.

        """


class IdentityReduce(Reduce):
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """Geneformer cell embedding function.

        Ignores expression embedding layer; uses embeddings based on identity embeddings.

        Args:
            gene_embedding_layer:  # TODO: fill out
            expression_embedding_layer: # TODO fill out

        """

        embeddings = gene_embedding_layer(identity_inputs)
        return embeddings


class SumReduce(Reduce):
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """ScGPT cell embedding callback.

        TODO: add "conditional tokens" (see Methods of https://www.nature.com/articles/s41592-024-02201-0#Sec14)

        Args:
            gene_embedding_layer:  # TODO: fill out
            expression_embedding_layer: # TODO fill out

        """
        # Convert str float_dtype -> actual torch dtype
        # torch_dtype = getattr(torch, self.float_dtype)
        # Cast expression_inputs to float_dtype
        expression_inputs = expression_inputs.to(torch.float32)

        partition = getattr(self.fc.data, "partition", "n/a")
        _check_embed_bounds("identity_inputs", identity_inputs, gene_embedding_layer, context=f"partition={partition}")
        _check_embed_bounds(
            "expression_inputs",
            expression_inputs,
            expression_embedding_layer,
            context=f"partition={partition}",
        )

        gene_embeddings = gene_embedding_layer(identity_inputs)
        expression_embeddings = expression_embedding_layer(expression_inputs)

        return gene_embeddings + expression_embeddings


class ChromosomeReduce(Reduce):
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cells using chromosome-aware sequences."""

        chrom_token_mask = identity_inputs < 0
        chrom_token_indices = identity_inputs[identity_inputs < 0]
        chrom_token_indices = -chrom_token_indices - self.fc.chrom_token_offset

        identity_inputs[chrom_token_mask] = 0

        gene_embeddings = gene_embedding_layer(identity_inputs)

        gene_embeddings[chrom_token_mask] = metadata_embedding_layer(chrom_token_indices)

        return gene_embeddings


class ChromosomeSumReduce(Reduce):
    def __call__(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
        metadata_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cells using chromosome-aware sequences."""

        chrom_token_mask = identity_inputs < 0
        chrom_token_indices = identity_inputs[identity_inputs < 0]
        chrom_token_indices = -chrom_token_indices - self.fc.chrom_token_offset

        identity_inputs[chrom_token_mask] = 0
        expression_inputs[chrom_token_mask] = 0

        gene_embeddings = gene_embedding_layer(identity_inputs)
        expression_embeddings = expression_embedding_layer(expression_inputs)
        meta_emb = metadata_embedding_layer(chrom_token_indices)

        meta_emb_gene = meta_emb.to(dtype=gene_embeddings.dtype, device=gene_embeddings.device)
        meta_emb_expr = meta_emb.to(dtype=expression_embeddings.dtype, device=expression_embeddings.device)

        gene_embeddings[chrom_token_mask] = meta_emb_gene
        expression_embeddings[chrom_token_mask] = meta_emb_expr

        return gene_embeddings + expression_embeddings
