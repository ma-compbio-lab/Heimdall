import torch

from Heimdall.losses import CLIPLoss, ContrastiveMulticlassLoss, ScheduledContrastiveLoss


class DummyTrainer:
    def __init__(self, step=0, total_training_steps=100):
        self.step = step
        self.total_training_steps = total_training_steps


def make_contrastive_views(batch_size=4, dim=8, scale=1.0):
    view_1 = torch.nn.functional.normalize(torch.randn(batch_size, dim), p=2, dim=1) * scale
    view_2 = torch.nn.functional.normalize(torch.randn(batch_size, dim), p=2, dim=1) * scale
    return torch.stack([view_1, view_2], dim=0)


def test_contrastive_multiclass_loss_returns_scalar():
    logits = make_contrastive_views(batch_size=4, dim=8)
    labels = torch.zeros(4, 8)

    loss = ContrastiveMulticlassLoss(trainer=DummyTrainer())(logits, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_clip_loss_returns_scalar():
    logits = make_contrastive_views(batch_size=2, dim=8)
    labels = torch.zeros(2, 4)

    loss = CLIPLoss(trainer=DummyTrainer())(logits, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_scheduled_contrastive_loss_switches_on_step():
    logits = make_contrastive_views(batch_size=3, dim=8, scale=2.0)
    labels = torch.zeros(3, 6)
    trainer = DummyTrainer(step=5, total_training_steps=100)
    loss_fn = ScheduledContrastiveLoss(trainer=trainer, switch_ratio=0.1)

    pre_switch_loss = loss_fn(logits, labels)
    expected_pre_switch = CLIPLoss(trainer=trainer)(logits, labels)
    assert torch.allclose(pre_switch_loss, expected_pre_switch)

    trainer.step = 10
    post_switch_loss = loss_fn(logits, labels)
    expected_post_switch = ContrastiveMulticlassLoss(trainer=trainer)(logits, labels)
    assert torch.allclose(post_switch_loss, expected_post_switch)
