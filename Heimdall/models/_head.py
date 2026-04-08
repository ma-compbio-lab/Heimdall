from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn, no_grad

from Heimdall.utils import project2simplex_


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    # predictions: torch.Tensor
    sequence_embeddings: torch.Tensor
    # pooled_embeddings: torch.Tensor
    cls_embeddings: torch.Tensor

    @property
    def device(self):
        return self.logits.device

    def to(self, device):
        for key, val in self.__dict__.items():
            self.__dict__[key] = val.to(device)

    @classmethod
    def reduce(cls, outputs: list["TransformerOutput"], reduction: Callable = torch.sum):
        keys = cls.__dict__["__annotations__"].keys()
        reduced_output = TransformerOutput(
            **{
                key: reduction(
                    torch.stack([getattr(output, key) for output in outputs], axis=0),
                    axis=0,
                )
                for key in keys
            },
        )
        return reduced_output

    def __post_init__(self):
        # ensure output tensors are in float32 format
        for k, v in self.__dict__.items():
            setattr(self, k, v.float())


class CellPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        cls_emb = encoder_output[:, 0, :]
        logits = self.decoder(cls_emb.unsqueeze(1)).squeeze(1)
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=encoder_output,
            cls_embeddings=cls_emb,
        )


class SeqPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        cls_emb = encoder_output[:, 0, :]
        logits = self.decoder(encoder_output[:, 1:, :])
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=encoder_output,
            cls_embeddings=cls_emb,
        )


class GenericPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        logits = self.decoder(encoder_output)
        logits = logits.squeeze(1)
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=logits,
            cls_embeddings=logits,
        )


class LinearDecoderMixin(nn.Module):
    def __init__(self, dim_in: int, dim_out: Optional[int] = None, dropout: float = 0.0, **kwargs):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.decoder = nn.Sequential(
            nn.Linear(dim_in, dim_out, **kwargs),
            nn.Dropout(dropout),
        )


class LinearCellPredHead(CellPredHeadMixin, LinearDecoderMixin):
    """Linear cell prediction head."""


class ExpressionOnlyCellPredHead(GenericPredHeadMixin, LinearDecoderMixin):
    """Logistic regression prediction head.

    Put expression be the input

    """


class LinearSeqPredHead(SeqPredHeadMixin, LinearDecoderMixin):
    """Linear sequence prediction head."""


class ContrastiveDecoder(nn.Module):
    """Return the two contrastive cell-view embeddings for loss construction."""

    def __init__(self, dim_in: int, dim_out: int | None = None, initial_tau_inv: float = 3.0):
        super().__init__()
        self.tau_inv = nn.Parameter(torch.tensor(initial_tau_inv), requires_grad=True)

    def forward(self, encoder_output):
        cls_embeddings = encoder_output[:, 0, :]
        double_batch_size, d_model = cls_embeddings.size()
        batch_size = double_batch_size // 2

        view_1 = cls_embeddings[:batch_size]
        view_2 = cls_embeddings[batch_size:]

        # Encode the learnable temperature directly into the view embeddings so
        # downstream losses can construct either contrastive matrix.
        scale = self.tau_inv.exp().sqrt()
        view_1 = torch.nn.functional.normalize(view_1, p=2, dim=1) * scale
        view_2 = torch.nn.functional.normalize(view_2, p=2, dim=1) * scale
        contrastive_views = torch.stack([view_1, view_2], dim=0)

        return TransformerOutput(
            logits=contrastive_views,
            sequence_embeddings=encoder_output,
            cls_embeddings=cls_embeddings,
        )


class SameViewContrastiveDecoder(nn.Module):
    """Deprecated alias for `ContrastiveDecoder`.

    Phase-specific contrastive matrices are now constructed inside the losses.

    """

    def __init__(self, dim_in: int, dim_out: int | None = None, initial_tau_inv: float = 3.0):
        super().__init__()
        self.contrastive_decoder = ContrastiveDecoder(
            dim_in=dim_in,
            dim_out=dim_out,
            initial_tau_inv=initial_tau_inv,
        )

    def forward(self, encoder_output):
        return self.contrastive_decoder(encoder_output)


class NonnegativeFactorizationMixin(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, k: int = 20, **kwargs):
        """Constrained Nonnegative Matrix Factorization (NMF) embedding.

        Args:
            dim_in: embedding size from previous layer's output
            dim_out: number of features in target output
            k: number of metafeatures

        """
        super().__init__()
        self.dim_reducer = nn.Linear(in_features=dim_in, out_features=k)
        self.sigmoid = nn.ReLU()
        self.metafeature_multiplier = nn.Linear(in_features=k, out_features=dim_out, bias=False)
        self.metafeatures = self.metafeature_multiplier.weight
        with no_grad():
            project2simplex_(self.metafeatures, dim=1)

        self.decoder = nn.Sequential(
            self.dim_reducer,
            self.sigmoid,
            self.metafeature_multiplier,
        )


class NonnegativeFactorizationPredHead(GenericPredHeadMixin, NonnegativeFactorizationMixin):
    """Nonnegative matrix factorization prediction head."""
