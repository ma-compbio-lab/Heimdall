import hydra
import numpy as np
import pytest
from omegaconf import OmegaConf

from Heimdall.task import ContrastiveViewTask, PairedInstanceTask, SingleInstanceTask, Tasklist


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


@pytest.fixture
def contrastive_representation(
    zero_expression_mock_dataset,
    zero_expression_identity_fg,
    zero_expression_identity_fe,
    geneformer_fc,
):
    zero_expression_mock_dataset._cfg = OmegaConf.create(
        {
            "scfm": {
                "trainer": {
                    "args": {
                        "batchsize": 4,
                    },
                },
            },
        },
    )
    zero_expression_mock_dataset.rng = np.random.default_rng(0)
    zero_expression_mock_dataset.set_representation_functions(
        fg=zero_expression_identity_fg,
        fe=zero_expression_identity_fe,
        fc=geneformer_fc,
    )
    return zero_expression_mock_dataset


def test_contrastive_task_instantiation(contrastive_representation):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/scfm/tasks"):
        conf = hydra.compose(config_name="contrast")
        OmegaConf.resolve(conf)

    task = ContrastiveViewTask(contrastive_representation, **conf.args)
    task.setup_labels()
    assert task.labels.shape == (contrastive_representation.adata.n_obs, 4)


def test_scheduled_contrastive_task_instantiation(contrastive_representation):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/scfm/tasks"):
        conf = hydra.compose(config_name="scheduled_contrast")
        OmegaConf.resolve(conf)

    task = ContrastiveViewTask(contrastive_representation, **conf.args)
    task.setup_labels()

    assert task.labels.shape == (contrastive_representation.adata.n_obs, 4)
    assert task.head_config.type == "Heimdall.models.ContrastiveDecoder"
    assert task.loss_config.type == "Heimdall.losses.ScheduledContrastiveLoss"
    assert task.loss_config.args.switch_ratio == 0.1


def test_contrastive_task_on_batch_samples_disjoint_panels(contrastive_representation):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/scfm/tasks"):
        conf = hydra.compose(config_name="contrast")
        OmegaConf.resolve(conf)

    task = ContrastiveViewTask(contrastive_representation, **conf.args)
    task.on_batch()

    assert len(task.panel_1_idx) >= conf.args.min_panel_size
    assert len(task.panel_2_idx) >= 1
    assert np.intersect1d(task.panel_1_idx, task.panel_2_idx).size == 0


def test_contrastive_task_get_inputs_restores_gene_panel(contrastive_representation):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/scfm/tasks"):
        conf = hydra.compose(config_name="contrast")
        OmegaConf.resolve(conf)

    task = ContrastiveViewTask(contrastive_representation, **conf.args)
    task.setup_labels()
    task.on_batch()

    shared_inputs = contrastive_representation.fc[0]
    shared_inputs["idx"] = 0
    inputs = task.get_inputs(0, shared_inputs)

    assert set(inputs) == {"identity_inputs", "expression_inputs", "expression_padding", "labels"}
    assert len(inputs["identity_inputs"]) == 2
    assert len(inputs["expression_inputs"]) == 2
    assert np.array_equal(
        contrastive_representation.fe.gene_panel_idx,
        np.arange(contrastive_representation.num_genes),
    )


def test_contrastive_task_collate_flattens_two_views(contrastive_representation):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config/scfm/tasks"):
        conf = hydra.compose(config_name="contrast")
        OmegaConf.resolve(conf)

    task = ContrastiveViewTask(contrastive_representation, **conf.args)
    task.setup_labels()
    task.on_batch()

    sample_0 = task.get_inputs(0, {**contrastive_representation.fc[0], "idx": 0})
    sample_1 = task.get_inputs(1, {**contrastive_representation.fc[1], "idx": 1})

    collated_identity = task.collate([sample_0["identity_inputs"], sample_1["identity_inputs"]])
    collated_labels = task.collate([sample_0["labels"], sample_1["labels"]])

    assert collated_identity.shape[0] == 4
    assert collated_labels.shape[0] == 4
