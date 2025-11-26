# <img src="https://github.com/ma-compbio/Heimdall/blob/workshop_prep/logo.png"  width=25%/>

[![Lint](https://github.com/gkrieg/Heimdall/actions/workflows/lint.yml/badge.svg)](https://github.com/gkrieg/Heimdall/actions/workflows/lint.yml)

Heimdall is a Python toolkit for implementing and benchmarking tokenizers for single-cell foundation models. Please cite our preprint if you use Heimdall in your work:

```latex
@article{haber2025heimdall,
  title={HEIMDALL: A Modular Framework for Tokenization in Single-Cell Foundation Models},
  author={Haber, Ellie and Alam, Shahul and Ho, Nicholas and Liu, Renming and Trop, Evan and Liang, Shaoheng and Yang, Muyu and Krieger, Spencer and Ma, Jian},
  journal={bioRxiv},
  pages={2025--11},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

Check out the documentation [here](https://heimdall-doc.readthedocs.io/en/latest/) for installation instructions, usage examples, and the complete API.


# Dev Notes

<details>
<summary>Click to expand dev instructions!</summary>

## Dev installation

```bash
pip install -r requirements.txt
```

Once the `pre-commit` command line tool is installed, every time you commit
some changes, it will perform several code-style checks and automatically
apply some fixes for you (if there is any issue). When auto-fixes
are applied, you need to recommit those changes. Note that this process can
take more than one round.

After you are done committing changes and are ready to push the commits to the
remote branch, run `nox` to perform a final quality check. Note that `nox` is
linting only and does not fix the issues for you. You need to address
the issues manually based on the instructions provided.

## Cheatsheet

```bash
# Run cell type classification dev experiment with wandb disabled
WANDB_MODE=disabled python train.py +experiments=cta_pancreas

# Run cell type classification dev experiment with wandb offline mode
WANDB_MODE=offline python train.py +experiments=cta_pancreas

# Run cell cell interaction dev experiment with wandb disabled
WANDB_MODE=disabled python train.py +experiments=cta_pancreas

# Run cell cell interaction dev experiment with wandb disabled and overwrite epochs
WANDB_MODE=disabled python train.py +experiments=cta_pancreas tasks.args.epochs=2

# Run cell cell interaction dev experiment with user profile (dev has wandb disabled by default)
python train.py +experiments=cta_pancreas user=lane-remy-dev
```

### Nox

Run code linting and unittests:

```bash
nox
```

Run dev experiments test on Lane compute node with CUDA (`lane-shared-dev` user profile):

```bash
nox -e test_experiments
```

Run fast dev experiments (only selected small datasets):

```bash
nox -e test_experiments -- quick_run
```

Run full dev experiments (including those datasets):

```bash
nox -e test_experiments -- full_run
```

Run dev experiments with a different user profile:

```bash
nox -e test_experiments -- user=box-remy-dev
```

## Local tests

We use [pytest](https://docs.pytest.org/en/stable/getting-started.html) to write local tests.
New test suites can be added under `tests/test_{suite_name}.py`.

Run a particular test suite with:

```bash
python -m pytest tests/test_{suite_name}.py
```

Run all tests but the integration test:

```bash
python -m pytest -m 'not integration'
```

Note: to run the integration test, you'll need to specify the Hydra user using a `.env` file.
The contents of the file should be like so:

```bash
HYDRA_USER=test
```

## Turning off caching

To turn off dataset caching for dev purposes,
set `cache_preprocessed_dataset_dir: null` in `config/global_vars.yaml`.
Alternatively, pass `cache_preprocessed_dataset_dir=null` through the command
line, e.g.,

```bash
python train.py +experiments=cta_pancreas cache_preprocessed_dataset_dir=null
```
