import hydra
import pytest
from omegaconf import OmegaConf

from Heimdall.task import PairedInstanceTask, SingleInstanceTask, Tasklist


def test_single_instance_task(mock_dataset):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/scfm/tasks"):
        conf = hydra.compose(
            config_name="spatial_cancer_split",
        )
        OmegaConf.resolve(conf)

    task = SingleInstanceTask(mock_dataset, **conf.args)


def test_paired_instance_task(mock_dataset):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/scfm/tasks"):
        conf = hydra.compose(
            config_name="reverse_perturbation",
        )
        OmegaConf.resolve(conf)

    task = PairedInstanceTask(mock_dataset, **conf.args)


def test_tasklist(mock_dataset):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "+scfm/tasks@scfm.tasks.1=spatial_cancer_split",
            ],
        )
        OmegaConf.resolve(conf)

    tasklist = Tasklist(
        mock_dataset,
        tasks=conf.scfm.tasks,
    )
    for _, subtask in tasklist:
        print(f"{subtask=}")


def test_multitask_tasklist(mock_dataset):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "+scfm/tasks@scfm.tasks.1=spatial_cancer_split",
                "+scfm/tasks@scfm.tasks.2=spatial_cancer_split",
            ],
        )
        OmegaConf.resolve(conf)

    tasklist = Tasklist(
        mock_dataset,
        tasks=conf.scfm.tasks,
    )
    for _, subtask in tasklist:
        print(f"{subtask=}")


def test_invalid_tasklist(mock_dataset):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "+scfm/tasks@scfm.tasks.1=spatial_cancer_split",
                "+scfm/tasks@scfm.tasks.2=new_sctab_split",
            ],
        )
        OmegaConf.resolve(conf)

    with pytest.raises(ValueError):
        tasklist = Tasklist(
            mock_dataset,
            tasks=conf.scfm.tasks,
        )
