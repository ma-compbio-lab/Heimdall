from pathlib import Path

from Heimdall.trainer import HeimdallTrainer


def test_get_latest_checkpoint_path(pytestconfig):
    results_folder = Path(pytestconfig.cache.mkdir("trainer_checkpoints"))
    milestone_path = results_folder / "milestone.txt"
    checkpoint_path = results_folder / "model-7.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    milestone_path.write_text("7")

    trainer = HeimdallTrainer.__new__(HeimdallTrainer)
    trainer.results_folder = results_folder
    trainer.initialize_checkpointing = lambda *args, **kwargs: None

    assert trainer.get_latest_checkpoint_path() == checkpoint_path
