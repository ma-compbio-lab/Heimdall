from pathlib import Path

import anndata as ad
import numpy as np
from omegaconf import OmegaConf
from pytest import fixture

from Heimdall.f_c import old_geneformer_fc
from Heimdall.f_g import IdentityFg


@fixture(scope="module")
def mock_dataset():
    gene_names = ["ENSG00000121410", "ENSG00000148584", "fake_gene", "ENSG00000175899"]

    mock_expression = np.array(
        [
            [1, 4, 3, 2],
            [2, 1, 4, 3],
            [3, 2, 1, 4],
            [4, 3, 2, 1],
        ],
    )

    mock_dataset = ad.AnnData(X=mock_expression)
    mock_dataset.var_names = gene_names

    return mock_dataset


@fixture(scope="module")
def identity_fg(mock_dataset):
    fg_config = OmegaConf.create(
        {
            "embedding_filepath": None,
            "d_embedding": 128,
        },
    )
    identity_fg = IdentityFg(mock_dataset, fg_config)

    return identity_fg


def test_old_geneformer_fc(mock_dataset, identity_fg):
    identity_fg.preprocess_embeddings()
    output = old_geneformer_fc(identity_fg, mock_dataset)

    expected = np.array(
        [
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
            [0, 1, 2, 3],
        ],
    )

    assert np.allclose(expected, output)
