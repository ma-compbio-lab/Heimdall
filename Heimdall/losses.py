import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class TrainerBoundMixin:
    def __init__(self, trainer, *args, **kwargs):
        self.trainer = trainer
        super().__init__(*args, **kwargs)

    @property
    def step(self):
        if self.trainer is None:
            return None
        return self.trainer.step


class MaskedLossMixin:
    def __init__(
        self,
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__(reduction="none", **kwargs)
        self._setup_reduction(reduction)

    def _setup_reduction(self, reduction):
        if reduction == "mean":
            self._reduce = self._reduce_mean
        elif reduction == "sum":
            self._reduce = self._reduce_sum
        else:
            raise ValueError(
                f"Unknown reduction option {reduction!r}. Available options are: 'mean', 'sum'",
            )

    def _reduce_mean(self, loss_mat: Tensor, mask: Tensor) -> Tensor:
        sizes = (~mask).sum(1, keepdim=True)
        return (loss_mat / sizes)[~mask].mean()

    def _reduce_sum(self, loss_mat: Tensor, mask: Tensor) -> Tensor:
        return (loss_mat[~mask] / loss_mat.shape[0]).sum()

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        mask = target.isnan()
        target[mask] = 0
        loss_mat = super().forward(input_, target)
        loss = self._reduce(loss_mat, mask)
        return loss


class MaskedBCEWithLogitsLoss(TrainerBoundMixin, MaskedLossMixin, nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss evaluated on unmasked entires."""


class TrainerMSELoss(TrainerBoundMixin, nn.MSELoss):
    """MSELoss that accepts Heimdall's injected `trainer` kwarg."""


class CrossEntropyFocalLoss(TrainerBoundMixin, nn.Module):
    def __init__(self, trainer, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float or list): Balancing factor for each class. If a single float, applies to class 1.
            gamma (float): Modulating factor to down-weight easy samples.
            reduction (str): 'none', 'mean', or 'sum'.
        """
        super().__init__(trainer=trainer)
        raise NotImplementedError("This class is not implemented correctly yet.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (not probabilities) with shape (batch_size, num_classes)
            targets: Class indices with shape (batch_size,)
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # Standard cross-entropy loss
        p_t = torch.exp(-ce_loss)  # Compute p_t = exp(-CE) (probability of the true class)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma) * ce_loss  # Apply focal loss formula

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss  # If reduction='none'


class FlattenMixin:
    def __init__(self, num_labels: int, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

    def forward(self, logits, labels):
        return super().forward(logits.view(-1, self.num_labels), labels.view(-1))


class FlattenCrossEntropyFocalLoss(FlattenMixin, CrossEntropyFocalLoss):
    """CrossEntropyFocalLoss with automatic flattening."""


class FlattenCrossEntropyLoss(TrainerBoundMixin, FlattenMixin, nn.CrossEntropyLoss):
    """CrossEntropyFocalLoss with automatic flattening."""


class ContrastiveMulticlassLoss(TrainerBoundMixin, nn.Module):
    """Cross-entropy over pairwise token matching across two cell views."""

    def __init__(self, trainer):
        super().__init__(trainer=trainer)
        self.cross_entropy = nn.CrossEntropyLoss()

    def _build_logits(self, logits: Tensor) -> Tensor:
        if logits.ndim == 2:
            return logits

        if logits.ndim != 3 or logits.size(0) != 2:
            raise ValueError(
                f"Expected stacked contrastive view embeddings with shape (2, batch, dim); got {tuple(logits.shape)}.",
            )

        view_1, view_2 = logits
        view_concat_1 = torch.cat([view_1, view_2], dim=0)
        view_concat_2 = torch.cat([view_2, view_1], dim=0)
        logits_both = torch.mm(view_concat_1, view_concat_2.t())
        logit_size = len(logits_both)
        non_trivial_mask = torch.roll(
            1 - torch.eye(logit_size, device=logits_both.device, dtype=logits_both.dtype),
            logit_size // 2,
            1,
        )
        return logits_both * non_trivial_mask

    def forward(self, logits, labels):
        logits = self._build_logits(logits)
        cells_per_batch = logits.size(0)
        target = torch.arange(cells_per_batch, device=logits.device)
        return self.cross_entropy(logits, target)


class CLIPLoss(TrainerBoundMixin, nn.Module):
    """Symmetric contrastive loss over both matching directions."""

    def __init__(self, trainer):
        super().__init__(trainer=trainer)
        self.contrastive_multiclass_loss = ContrastiveMulticlassLoss(trainer=trainer)

    def _build_logits(self, logits: Tensor) -> Tensor:
        if logits.ndim == 2:
            return logits

        if logits.ndim != 3 or logits.size(0) != 2:
            raise ValueError(
                f"Expected stacked contrastive view embeddings with shape (2, batch, dim); got {tuple(logits.shape)}.",
            )

        view_1, view_2 = logits
        return torch.mm(view_1, view_2.t())

    def forward(self, logits, labels):
        logits = self._build_logits(logits)
        loss_1 = self.contrastive_multiclass_loss(logits, labels)

        transposed_logits = logits.t().contiguous()
        loss_2 = self.contrastive_multiclass_loss(transposed_logits, labels)

        return (loss_1 + loss_2) / 2


class ScheduledContrastiveLoss(TrainerBoundMixin, nn.Module):
    """Switch between two contrastive objectives based on the trainer step."""

    def __init__(
        self,
        trainer,
        switch_step: int | None = None,
        switch_ratio: float | None = None,
    ):
        super().__init__(trainer=trainer)
        if (switch_step is None) == (switch_ratio is None):
            raise ValueError("Exactly one of `switch_step` or `switch_ratio` must be provided.")
        if switch_ratio is not None and not (0.0 <= switch_ratio <= 1.0):
            raise ValueError("`switch_ratio` must lie in [0, 1].")

        self._switch_step = switch_step
        self.switch_ratio = switch_ratio
        self.pre_switch_loss = CLIPLoss(trainer=trainer)
        self.post_switch_loss = ContrastiveMulticlassLoss(trainer=trainer)

    @property
    def switch_step(self):
        if self._switch_step is None:
            self._switch_step = int(self.trainer.total_training_steps * self.switch_ratio)
        return self._switch_step

    @property
    def active_loss(self):
        if self.step is None or self.step < self.switch_step:
            return self.pre_switch_loss
        return self.post_switch_loss

    def forward(self, logits, labels):
        return self.active_loss(logits, labels)
