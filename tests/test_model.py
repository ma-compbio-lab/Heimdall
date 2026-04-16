import os
from textwrap import dedent, indent

import anndata as ad
import hydra
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from omegaconf import OmegaConf, open_dict
from pytest import fixture

from Heimdall.cell_representations import CellRepresentation, setup_data
from Heimdall.models import setup_model
from Heimdall.utils import INPUT_KEYS, get_collation_closure, get_dtype, instantiate_from_config

load_dotenv()

model_configs = ["transformer", "expression_weighted_sum", "average"]
fc_configs = ["geneformer", "geneformer", "geneformer"]


@fixture(scope="module")
def paired_task_config(request, toy_paired_data_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "scfm/model=transformer",
                "scfm.model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                "scfm/dataset=test",
                "scfm/trainer=paired",
                "+scfm/tasks@scfm.tasks.default=paired_test",
                f"scfm.tasks.default.args.reducer_config.type={request.param}",
                f"scfm.dataset.preprocess_args.data_path={toy_paired_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "scfm/fg=identity",
                "scfm/fe=2nn",
                "scfm/fc=test",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def single_task_config(request, toy_single_data_path):
    model_config, fc_config = request.param
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"scfm/model={model_config}",
                "scfm.model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                "scfm/dataset=test",
                "+scfm/tasks@scfm.tasks.default=test",
                f"scfm.dataset.preprocess_args.data_path={toy_single_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "scfm/fg=identity",
                "scfm/fe=zero",
                f"scfm/fc={fc_config}",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def multitask_config(request, toy_single_data_path):
    model_config, fc_config = request.param
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"scfm/model={model_config}",
                "scfm.model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                "scfm/dataset=test",
                "+scfm/tasks@scfm.tasks.1=test",
                "+scfm/tasks@scfm.tasks.2=test",
                f"scfm.dataset.preprocess_args.data_path={toy_single_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "scfm/fg=identity",
                "scfm/fe=zero",
                f"scfm/fc={fc_config}",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def expression_only_config(request, toy_single_data_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "scfm/model=expression_only",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                "scfm/dataset=test",
                "+scfm/tasks@scfm.tasks.default=test",
                f"scfm.dataset.preprocess_args.data_path={toy_single_data_path}",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "scfm/fg=dummy",
                "scfm/fe=dummy",
                "scfm/fc=dummy",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def autoencoder_config(toy_single_data_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "scfm/model=expression_only",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                "scfm/dataset=test",
                "+scfm/tasks@scfm.tasks.default=autoencoder",
                f"scfm.dataset.preprocess_args.data_path={toy_single_data_path}",
                "cache_preprocessed_dataset_dir=null",
                "work_dir=work_dir",
                "scfm/fg=dummy",
                "scfm/fe=dummy",
                "scfm/fc=dummy",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def bottleneck_autoencoder_config(toy_single_data_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "scfm/model=bottleneck",
                "scfm.model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                "scfm/dataset=test",
                "+scfm/tasks@scfm.tasks.default=autoencoder",
                f"scfm.dataset.preprocess_args.data_path={toy_single_data_path}",
                "cache_preprocessed_dataset_dir=null",
                "work_dir=work_dir",
                "scfm/fg=dummy",
                "scfm/fe=dummy",
                "scfm/fc=dummy",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def partition_config(request, toy_partitioned_data_path):
    model_config, fc_config = request.param
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                f"scfm/model={model_config}",
                "scfm.model.args.d_model=128",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                "scfm/dataset=test",
                "+scfm/tasks@scfm.tasks.default=test",
                f"scfm.dataset.preprocess_args.data_path={toy_partitioned_data_path}",
                "scfm/trainer=partitioned",
                f"cache_preprocessed_dataset_dir=null",
                f"work_dir=work_dir",
                "scfm/fg=identity",
                "scfm/fe=zero",
                f"scfm/fc={fc_config}",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


@fixture(scope="module")
def contrastive_task_config(request, toy_single_data_path):
    with hydra.initialize(version_base=None, config_path="../Heimdall/config"):
        conf = hydra.compose(
            config_name="config",
            overrides=[
                "scfm/model=transformer",
                "scfm.model.args.d_model=64",
                f"data_path={os.environ['DATA_PATH']}",
                f"ensembl_dir={os.environ['DATA_PATH']}",
                "scfm/dataset=test",
                f"+scfm/tasks@scfm.tasks.default={request.param}",
                f"scfm.dataset.preprocess_args.data_path={toy_single_data_path}",
                "cache_preprocessed_dataset_dir=null",
                "work_dir=work_dir",
                "run_wandb=false",
                "scfm/fg=identity",
                "scfm/fe=zero",
                "scfm/fc=geneformer",
                "scfm.trainer.args.batchsize=4",
            ],
        )
        OmegaConf.resolve(conf)

    return conf


# @fixture(scope="module")
# def accelerator(config):
#     setup
def instantiate_and_run_model(config):
    accelerator, cr, run_wandb, only_preprocess_data = setup_data(config)

    if only_preprocess_data:
        return

    model = setup_model(config, cr, is_main_process=accelerator.is_main_process)

    # Avoid the dataloader worker stack for model smoke tests.
    batch = get_collation_closure(cr.tasklist)([cr.datasets["train"][0]])
    inputs = {input_key: batch[input_key] for input_key in INPUT_KEYS if input_key in batch}
    model(inputs=inputs)


@pytest.mark.parametrize(
    "single_task_config",
    zip(model_configs, fc_configs),
    indirect=True,
)
def test_single_task_model_instantiation(single_task_config):
    instantiate_and_run_model(single_task_config)


@pytest.mark.parametrize(
    "multitask_config",
    zip(model_configs, fc_configs),
    indirect=True,
)
def test_multitask_model_instantiation(multitask_config):
    instantiate_and_run_model(multitask_config)


def test_expression_only_instantiation(expression_only_config):
    instantiate_and_run_model(expression_only_config)


def test_autoencoder_task_instantiation(autoencoder_config):
    instantiate_and_run_model(autoencoder_config)


def test_bottleneck_autoencoder_task_instantiation(bottleneck_autoencoder_config):
    instantiate_and_run_model(bottleneck_autoencoder_config)


@pytest.mark.parametrize(
    "paired_task_config",
    [
        "Heimdall.models.SumReducer",
        "Heimdall.models.MeanReducer",
        "Heimdall.models.SymmetricConcatReducer",
        "Heimdall.models.AsymmetricConcatReducer",
    ],
    indirect=True,
)
def test_paired_task_model_instantiation(paired_task_config):
    instantiate_and_run_model(paired_task_config)


@pytest.mark.parametrize(
    "partition_config",
    zip(model_configs, fc_configs),
    indirect=True,
)
def test_partitioned_model_instantiation(partition_config):
    instantiate_and_run_model(partition_config)


@pytest.mark.parametrize(
    "contrastive_task_config",
    ["contrast", "same_view_contrast"],
    indirect=True,
)
def test_contrastive_task_model_instantiation(contrastive_task_config):
    instantiate_and_run_model(contrastive_task_config)
