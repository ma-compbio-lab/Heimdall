import numpy as np
from scipy.sparse import csr_array

from Heimdall.fe import IdentityFe
from Heimdall.fg import IdentityFg
from Heimdall.tokenizer import KnnTokenizerContext, TokenizerContext


def test_default_tokenizer_context_uses_fg_state_without_touching_adata_var(mock_dataset, identity_fg_config):
    context = TokenizerContext(adata=mock_dataset.adata, raw_gene_names=mock_dataset.raw_gene_names)
    fg = IdentityFg(context, **identity_fg_config)
    context.set_representation_functions(fg=fg)
    fg.preprocess_embeddings()

    assert np.array_equal(context.gene_names, mock_dataset.raw_gene_names)

    fg.identity_valid_mask = np.array([True, False, True, True], dtype=bool)

    assert np.array_equal(context.identity_valid_mask, np.array([True, False, True, True], dtype=bool))
    assert np.array_equal(context.gene_names, mock_dataset.raw_gene_names[[0, 2, 3]])
    assert "identity_valid_mask" not in mock_dataset.adata.var.columns
    assert "identity_embedding_index" not in mock_dataset.adata.var.columns


def test_knn_tokenizer_context_uses_dataframe_columns_and_feature_matrix(
    mock_dataset,
    knn_feature_frame,
    identity_fg_config,
    identity_fe_config,
):
    feature_frame = knn_feature_frame
    mock_dataset.adata.obsm["X_atlas_knn_mean_expr"] = feature_frame

    context = KnnTokenizerContext(adata=mock_dataset.adata, raw_gene_names=mock_dataset.raw_gene_names)
    assert np.array_equal(context.raw_gene_names, np.array(["g1", "g2", "g3", "g4"]))
    assert np.allclose(context.feature_matrix, feature_frame.to_numpy())

    fg = IdentityFg(context, **identity_fg_config)
    fe = IdentityFe(context, drop_zeros=False, **identity_fe_config)
    context.set_representation_functions(fg=fg, fe=fe)
    fg.preprocess_embeddings()
    fe.preprocess_embeddings()
    fg.identity_valid_mask = np.array([True, False, True, True], dtype=bool)

    identity_inputs, expression_inputs = fe[1]

    assert np.array_equal(context.gene_names, np.array(["g1", "g3", "g4"]))
    assert np.array_equal(identity_inputs, np.array([0, 1, 2]))
    assert np.allclose(expression_inputs, np.array([20.0, 0.0, 23.0], dtype=np.float64))
    assert "identity_valid_mask" not in mock_dataset.adata.var.columns
    assert "identity_embedding_index" not in mock_dataset.adata.var.columns


def test_knn_tokenizer_context_uses_uns_feature_gene_names_with_array_feature_matrix(
    mock_dataset,
    identity_fg_config,
    identity_fe_config,
):
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

    context = KnnTokenizerContext(adata=mock_dataset.adata, raw_gene_names=mock_dataset.raw_gene_names)
    assert np.array_equal(context.raw_gene_names, np.array(["g1", "g2", "g3", "g4"]))
    assert np.allclose(context.feature_matrix.toarray(), feature_matrix.toarray())

    fg = IdentityFg(context, **identity_fg_config)
    fe = IdentityFe(context, **identity_fe_config)
    context.set_representation_functions(fg=fg, fe=fe)
    fg.preprocess_embeddings()
    fe.preprocess_embeddings()
    fg.identity_valid_mask = np.array([True, False, True, True], dtype=bool)

    identity_inputs, expression_inputs = fe[1]

    assert np.array_equal(context.gene_names, np.array(["g1", "g3", "g4"]))
    assert np.array_equal(identity_inputs, np.array([0, 1, 2]))
    assert np.allclose(expression_inputs, np.array([20.0, 22.0, 23.0], dtype=np.float64))


def test_cell_representation_gene_names_delegate_to_tokenizer_context(mock_dataset, identity_fg_config):
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

    context = KnnTokenizerContext(adata=mock_dataset.adata, raw_gene_names=mock_dataset.raw_gene_names)
    fg = IdentityFg(context, **identity_fg_config)
    context.set_representation_functions(fg=fg)
    fg.preprocess_embeddings()
    fg.identity_valid_mask = np.array([True, False, True, True], dtype=bool)

    mock_dataset.tokenizer_context = context
    mock_dataset.fg = fg

    assert np.array_equal(mock_dataset.gene_names, np.array(["g1", "g3", "g4"]))
    assert mock_dataset.num_genes == 3
