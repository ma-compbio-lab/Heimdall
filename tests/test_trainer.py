from pathlib import Path

import hydra

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


def test_checkpoint_directory_depends_on_scfm_tasks(tmp_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        contrast_config = hydra.compose(
            config_name="config",
            overrides=[
                "work_dir=null",
                f"cache_preprocessed_dataset_dir={tmp_path}",
                "scfm/model=transformer",
                "scfm/dataset=test",
                "scfm/fg=identity",
                "scfm/fe=zero",
                "scfm/fc=geneformer",
                "+scfm/tasks@scfm.tasks.1=contrast",
            ],
        )
        multitask_config = hydra.compose(
            config_name="config",
            overrides=[
                "work_dir=null",
                f"cache_preprocessed_dataset_dir={tmp_path}",
                "scfm/model=transformer",
                "scfm/dataset=test",
                "scfm/fg=identity",
                "scfm/fe=zero",
                "scfm/fc=geneformer",
                "+scfm/tasks@scfm.tasks.1=mlm",
                "+scfm/tasks@scfm.tasks.2=contrast",
            ],
        )

    contrast_trainer = HeimdallTrainer.__new__(HeimdallTrainer)
    contrast_trainer.cfg = contrast_config

    multitask_trainer = HeimdallTrainer.__new__(HeimdallTrainer)
    multitask_trainer.cfg = multitask_config

    contrast_checkpoint_dir = contrast_trainer.get_checkpoint_directory()
    multitask_checkpoint_dir = multitask_trainer.get_checkpoint_directory()

    assert contrast_checkpoint_dir != multitask_checkpoint_dir
