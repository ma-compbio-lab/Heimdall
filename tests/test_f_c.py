from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from Heimdall.f_c import old_geneformer_fc


@pytest.fixture(scope="module")
def mock_fg():
    num_genes = 3
    f_g = {f"gene_{index}": index for index in range(num_genes)}

    return f_g


def test_old_geneformer_fc(mock_fg):
    gene_names = list(mock_fg.keys())
    mock_expression = np.array(
        [
            [1, 3, 2],
            [2, 1, 3],
            [3, 2, 1],
        ],
    )

    mock_dataset = ad.AnnData(X=mock_expression)
    mock_dataset.var_names = gene_names

    output = old_geneformer_fc(mock_fg, mock_dataset)

    # TODO: I'm not sure about this... it seems like the current implementation doesn't break
    # sorting ties consistently?
    expected = np.array(
        [
            [1, 2, 0],
            [2, 0, 1],
            [0, 1, 2],
        ],
    )

    assert np.allclose(expected, output)
