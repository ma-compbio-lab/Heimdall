from omegaconf import OmegaConf
from pytest import fixture

from Heimdall.cell_representations import CellRepresentation
from Heimdall.models import HeimdallModel


@fixture(scope="module")
def paired_task_config():
    config_string = """
    project_name: Cell_Cell_interaction
    run_name: run_name
    work_dir: work_dir
    seed: 42
    data_path: /work/magroup/scllm/datasets
    ensembl_dir: /work/magroup/shared/Heimdall/data/gene_mapping
    cache_preprocessed_dataset_dir: /work/magroup/shared/Heimdall/preprocessed_data
    entity: Heimdall
    model:
      type: transformer
      args:
        d_model: 128
        pos_enc: BERT
        num_encoder_layers: 2
        nhead: 2
        hidden_act: gelu
        hidden_dropout_prob: 0.1
        attention_probs_dropout_prob: 0.1
        use_flash_attn: false
        pooling: cls_pooling
    dataset:
      dataset_name: zeng_merfish_ccc_subset
      preprocess_args:
        data_path: /work/magroup/shared/Heimdall/data/cell_to_cell_interaction/adata_MERFISH_Zeng_Slices1to10.h5ad
        top_n_genes: 1000
        normalize: true
        log_1p: true
        scale_data: true
        species: mouse
    tasks:
      args:
        task_type: binary
        interaction_type: _all_
        label_col_name: class
        splits:
          type: predefined
          keys_:
            train: train
            val: val
            test: test
        metrics:
        - Accuracy
        shuffle: true
        batchsize: 32
        epochs: 10
        prediction_dim: 14
        dataset_config:
          type: Heimdall.datasets.PairedInstanceDataset
        head_config:
          type: Heimdall.models.LinearCellPredHead
          args: null
    scheduler:
      name: cosine
      lr_schedule_type: cosine
      warmup_ratio: 0.1
      num_epochs: 20
    trainer:
      accelerator: cuda
      precision: 32-true
      random_seed: 42
      per_device_batch_size: 64
      accumulate_grad_batches: 1
      grad_norm_clip: 1.0
      fastdev: false
    optimizer:
      name: AdamW
      args:
        lr: 0.0001
        weight_decay: 0.1
        betas:
        - 0.9
        - 0.95
        foreach: false
    f_c:
      type: Heimdall.f_c.GeneformerFc
      args:
        max_input_length: 128
    fe:
      type: Heimdall.fe.SortingFe
      args:
        embedding_filepath: null
        num_embeddings: null
        d_embedding: 128
    f_g:
      name: IdentityFg
      type: Heimdall.f_g.IdentityFg
      args:
        embedding_filepath: null
        d_embedding: 128
    loss:
      name: CrossEntropyLoss
    """
    conf = OmegaConf.create(config_string)

    return conf


@fixture(scope="module")
def single_task_config():
    config_string = """
    project_name: Cell_Type_Classification_dev
    run_name: run_name
    work_dir: work_dir
    seed: 42
    data_path: /work/magroup/scllm/datasets
    ensembl_dir: /work/magroup/shared/Heimdall/data/gene_mapping
    cache_preprocessed_dataset_dir: /work/magroup/shared/Heimdall/preprocessed_data
    entity: Heimdall
    model:
      type: transformer
      args:
        d_model: 128
        pos_enc: BERT
        num_encoder_layers: 2
        nhead: 2
        hidden_act: gelu
        hidden_dropout_prob: 0.1
        attention_probs_dropout_prob: 0.1
        use_flash_attn: false
        pooling: cls_pooling
    dataset:
      dataset_name: cell_type_classification
      preprocess_args:
        data_path: /work/magroup/scllm/datasets/sc_sub_nick.h5ad
        top_n_genes: 1000
        normalize: true
        log_1p: true
        scale_data: true
        species: mouse
    tasks:
      args:
        task_type: multiclass
        label_col_name: class
        metrics:
        - Accuracy
        - MatthewsCorrCoef
        train_split: 0.8
        shuffle: true
        batchsize: 32
        epochs: 10
        prediction_dim: 14
        dataset_config:
          type: Heimdall.datasets.SingleInstanceDataset
        head_config:
          type: Heimdall.models.LinearCellPredHead
          args: null
    scheduler:
      name: cosine
      lr_schedule_type: cosine
      warmup_ratio: 0.1
      num_epochs: 20
    trainer:
      accelerator: cuda
      precision: 32-true
      random_seed: 42
      per_device_batch_size: 64
      accumulate_grad_batches: 1
      grad_norm_clip: 1.0
      fastdev: false
    optimizer:
      name: AdamW
      args:
        lr: 0.0001
        weight_decay: 0.1
        betas:
        - 0.9
        - 0.95
        foreach: false
    f_c:
      type: Heimdall.f_c.GeneformerFc
      args:
        max_input_length: 128
    fe:
      type: Heimdall.fe.SortingFe
      args:
        embedding_filepath: null
        num_embeddings: null
        d_embedding: 128
    f_g:
      name: IdentityFg
      type: Heimdall.f_g.IdentityFg
      args:
        embedding_filepath: null
        d_embedding: 128
    loss:
      name: CrossEntropyLoss
    """
    conf = OmegaConf.create(config_string)

    return conf


def test_single_transformer_instantiation(single_task_config):
    cr = CellRepresentation(single_task_config)  # takes in the whole config from hydra

    model = HeimdallModel(
        data=cr,
        model_config=single_task_config.model.args,
        task_config=single_task_config.tasks.args,
    )


def test_paired_transformer_instantiation(paired_task_config):
    cr = CellRepresentation(paired_task_config)  # takes in the whole config from hydra

    model = HeimdallModel(
        data=cr,
        model_config=paired_task_config.model.args,
        task_config=paired_task_config.tasks.args,
    )
