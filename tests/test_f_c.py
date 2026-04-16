from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pytest import fixture
from scipy.sparse import csr_array

from Heimdall.fc import Fc
from Heimdall.fe import IdentityFe
from Heimdall.fg import IdentityFg
from Heimdall.tokenizer import KnnTokenizerContext


def test_dummy_getitem(geneformer_fc, scgpt_fc):
    dummy_index = -1

    outputs = geneformer_fc[dummy_index]
    assert np.all(outputs["identity_inputs"] == geneformer_fc.fg.pad_value)
    assert np.all(outputs["expression_inputs"] == geneformer_fc.fe.pad_value)
    assert np.all(outputs["expression_padding"])

    outputs = scgpt_fc[dummy_index]
    assert np.all(outputs["identity_inputs"] == scgpt_fc.fg.pad_value)
    assert np.all(outputs["expression_inputs"] == scgpt_fc.fe.pad_value)
    assert np.all(outputs["expression_padding"])


def test_geneformer_fc_preprocess_cells_and_getitem(zero_expression_mock_dataset, geneformer_fc):
    zero_expression_mock_dataset.set_representation_functions(
        fg=geneformer_fc.fg,
        fe=geneformer_fc.fe,
        fc=geneformer_fc,
    )

    identity_expected = csr_array(
        np.array(
            [
                [1, 2, 3],
                [2, 3, 0],
                [3, 0, 1],
                [0, 1, 2],
            ],
        ),
    )

    _, raw_seq_length = identity_expected.shape

    for cell_index in range(len(zero_expression_mock_dataset.adata)):
        outputs = geneformer_fc[cell_index]
        assert np.allclose(identity_expected[[cell_index], :].toarray(), outputs["identity_inputs"][:raw_seq_length])
        assert len(outputs["identity_inputs"]) == geneformer_fc.max_input_length

        assert not np.any(outputs["expression_padding"][:raw_seq_length])
        assert np.all(outputs["expression_padding"][raw_seq_length:])


def test_scgpt_fc_preprocess_cells_and_getitem(zero_expression_mock_dataset, scgpt_fc):
    zero_expression_mock_dataset.set_representation_functions(
        fg=scgpt_fc.fg,
        fe=scgpt_fc.fe,
        fc=scgpt_fc,
    )

    identity_expected = csr_array(
        np.array(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
        ),
    )

    expression_expected = csr_array(
        np.array(
            [
                [3, 2, 1],
                [1, 3, 2],
                [2, 1, 3],
                [3, 2, 1],
            ],
        ),
    )

    _, raw_seq_length = identity_expected.shape

    seed = 0
    rng = np.random.default_rng(seed)
    for cell_index in range(len(zero_expression_mock_dataset.adata)):
        outputs = scgpt_fc[cell_index]
        sample_indices = rng.choice(raw_seq_length, raw_seq_length, replace=False)
        assert np.allclose(identity_expected[[cell_index], sample_indices], outputs["identity_inputs"])
        assert np.allclose(expression_expected[[cell_index], sample_indices], outputs["expression_inputs"])
        assert len(outputs["identity_inputs"]) == scgpt_fc.max_input_length

        assert not np.any(outputs["expression_padding"][: scgpt_fc.max_input_length])


def test_scBERT_fc_preprocess_cells_and_getitem(zero_expression_mock_dataset, scbert_fc):
    zero_expression_mock_dataset.set_representation_functions(
        fg=scbert_fc.fg,
        fe=scbert_fc.fe,
        fc=scbert_fc,
    )
    identity_expected = csr_array(
        np.array(
            [
                [1, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 2],
            ],
        ),
    )

    expression_expected = csr_array(
        np.array(
            [
                [2, 2, 1],
                [1, 2, 2],
                [2, 1, 2],
                [2, 2, 1],
            ],
        ),
    )

    _, raw_seq_length = identity_expected.shape

    seed = 0
    rng = np.random.default_rng(seed)
    for cell_index in range(len(zero_expression_mock_dataset.adata)):
        outputs = scbert_fc[cell_index]
        sample_indices = rng.choice(raw_seq_length, raw_seq_length, replace=False)
        assert np.allclose(identity_expected[[cell_index], sample_indices], outputs["identity_inputs"])
        assert np.allclose(expression_expected[[cell_index], sample_indices], outputs["expression_inputs"])
        assert len(outputs["identity_inputs"]) == scbert_fc.max_input_length

        assert not np.any(outputs["expression_padding"][: scbert_fc.max_input_length])


# def test_uce_fc_order(mock_dataset_all_valid_genes, uce_fc):
#     _, num_genes = weighted_sampling_fe.adata.shape
#
#     expected = np.array(
#         [
#             [1, 1, 0, 0, 2],
#             [2, 2, 2, 1, 2],
#             [2, 0, 2, 0, 2],
#             [0, 2, 1, 0, 0],
#         ],
#     )
#
#     for cell_index in range(len(identity_fg.adata)):
#         cell_outputs["identity_inputs"], cell_expression_inputs = weighted_sampling_fe[cell_index]
#         assert np.allclose(expected[cell_index], cell_outputs["identity_inputs"])
#
#     assert weighted_sampling_fe.pad_value == 4
#     assert weighted_sampling_fe.mask_value == 5


def test_uce_fc_preprocess_cells_and_getitem(mock_dataset_all_valid_genes, uce_fc):
    mock_dataset_all_valid_genes.set_representation_functions(
        fg=uce_fc.fg,
        fe=uce_fc.fe,
        fc=uce_fc,
    )

    identity_expected = csr_array(
        np.array(
            [
                [-4, 0, 0],
                [-2, 1, -1],
                [-2, 1, 1],
                [-4, 0, 0],
            ],
        ),
    )

    expression_expected = csr_array(
        np.array(
            [
                [-4, 1, 1],
                [-2, 1, -1],
                [-2, 2, 2],
                [-4, 4, 4],
            ],
        ),
    )

    _, raw_seq_length = identity_expected.shape

    for cell_index in range(len(mock_dataset_all_valid_genes.adata)):
        outputs = uce_fc[cell_index]
        assert np.allclose(identity_expected[[cell_index], :].toarray(), outputs["identity_inputs"][:raw_seq_length])
        assert np.allclose(
            expression_expected[[cell_index], :].toarray(),
            outputs["expression_inputs"][:raw_seq_length],
        )
        assert len(outputs["identity_inputs"]) == uce_fc.max_input_length

        assert not np.any(outputs["expression_padding"][:raw_seq_length])


def test_geneformer_fc_reduce(geneformer_fc):
    ...
    # geneformer_fc.reduce() # TODO: fill out function call

    # output = mock_dataset.obsm["cell_outputs["identity_inputs"]"]

    # expected = np.array(
    #     [
    #         [1, 2, 3, 0],
    #         [2, 3, 0, 1],
    #         [3, 0, 1, 2],
    #         [0, 1, 2, 3],
    #     ],
    # )

    # assert np.allclose(expected, output)


def test_knn_context_with_default_fc_reads_precomputed_expression(mock_dataset, identity_fg_config, identity_fe_config):
    feature_matrix = csr_array(
        np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
                [40.0, 41.0, 42.0, 43.0],
            ],
            dtype=np.float32,
        ),
    )
    mock_dataset.adata.obsm["X_atlas_knn_mean_expr"] = feature_matrix
    mock_dataset.adata.uns["atlas_knn"] = {"feature_gene_names": ["g1", "g2", "g3", "g4"]}

    context = KnnTokenizerContext(
        adata=mock_dataset.adata,
        raw_gene_names=mock_dataset.raw_gene_names,
        feature_obsm_key="X_atlas_knn_mean_expr",
    )
    fg = IdentityFg(context, **identity_fg_config)
    fe = IdentityFe(context, **identity_fe_config)
    fc = Fc(
        fg=fg,
        fe=fe,
        context=context,
        max_input_length=4,
        float_dtype="float32",
        rng=0,
        embedding_parameters=OmegaConf.create({"type": "torch.nn.Module"}),
        tailor_config=OmegaConf.create({"type": "Heimdall.tailor.ReorderTailor"}),
        order_config=OmegaConf.create({"type": "Heimdall.order.ExpressionOrder"}),
        reduce_config=OmegaConf.create({"type": "Heimdall.reduce.IdentityReduce"}),
    )
    context.set_representation_functions(fg=fg, fe=fe, fc=fc)
    fg.preprocess_embeddings()
    fe.preprocess_embeddings()

    outputs = fc[1]

    assert np.array_equal(context.gene_names, np.array(["g1", "g2", "g3", "g4"]))
    assert outputs["identity_inputs"].shape == (4,)
    assert outputs["expression_padding"].shape == (4,)
    assert np.allclose(outputs["expression_inputs"], np.sort(feature_matrix.toarray()[1])[::-1])
