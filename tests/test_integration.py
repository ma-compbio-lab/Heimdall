import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import hydra
import pytest
from dotenv import load_dotenv
from omegaconf import OmegaConf, open_dict

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel
from Heimdall.trainer import HeimdallTrainer
from Heimdall.utils import count_parameters, get_dtype, instantiate_from_config

load_dotenv()

if "HYDRA_USER" not in os.environ:
    pytest.skip(".env file must specify HYDRA_USER for integrated test.", allow_module_level=True)


@pytest.mark.integration
def test_default_hydra_train():
    with hydra.initialize(version_base=None, config_path="../config"):
        config = hydra.compose(
            config_name="config",
            overrides=[
                # "+experiments_dev=classification_experiment_dev",
                "+experiments=spatial_cancer_split1",
                # "user=lane-nick"
                "model=transformer",
                "fg=pca_esm2",
                "fe=identity",
                "fc=uce",
                "seed=55",
                "project_name=demo",
                "tasks.args.epochs=1",
                "fc.args.max_input_length=512",
                "fc.args.tailor_config.args.sample_size=450",
                # f"user={os.environ['HYDRA_USER']}"
            ],
        )
        print(OmegaConf.to_yaml(config))

    with open_dict(config):
        only_preprocess_data = config.pop("only_preprocess_data", None)
        # pop so hash of cfg is not changed depending on value

    cr = instantiate_from_config(config.tasks.args.cell_rep_config, config)

    if only_preprocess_data:
        return

    float_dtype = get_dtype(config.float_dtype)

    model = HeimdallModel(
        data=cr,
        model_config=config.model,
        task_config=config.tasks.args,
    ).to(float_dtype)

    num_params = count_parameters(model)

    print(f"\nModel constructed:\n{model}\nNumber of trainable parameters {num_params:,}\n")

    trainer = HeimdallTrainer(cfg=config, model=model, data=cr, run_wandb=False)

    trainer.fit(resume_from_checkpoint=False, checkpoint_every_n_epochs=20)

    valid_log, _ = trainer.validate_model(trainer.dataloader_val, dataset_type="valid")

    assert valid_log["valid_MatthewsCorrCoef"] > 0.25
