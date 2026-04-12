"""Evaluation helpers for Heimdall models."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate
from tqdm.auto import tqdm

from Heimdall.utils import FC_KEYS


def _masked_mean(sequence_embeddings, attention_mask):
    if attention_mask is None:
        return sequence_embeddings.mean(dim=1)

    valid_mask = ~attention_mask.bool()
    expanded_mask = valid_mask.unsqueeze(-1)
    masked_embeddings = sequence_embeddings * expanded_mask
    valid_counts = expanded_mask.sum(dim=1).clamp(min=1)
    return masked_embeddings.sum(dim=1) / valid_counts


def pool_sequence_tokens(trainer, sequence_embeddings, attention_mask):
    pooling = getattr(trainer.model_cfg.args, "pooling", "mean_pooling")

    if pooling == "mean_pooling":
        if sequence_embeddings.ndim == 2:
            return sequence_embeddings
        return _masked_mean(sequence_embeddings, attention_mask)

    if pooling == "cls_pooling":
        if sequence_embeddings.ndim != 3:
            raise ValueError(
                "Expected token-level encoder outputs for `cls_pooling`, "
                f"but received shape {tuple(sequence_embeddings.shape)}.",
            )
        return _masked_mean(sequence_embeddings[:, 1:, :], attention_mask)

    raise ValueError(f"Unsupported pooling mode for embedding export: {pooling}")


def collate_fc_outputs(fc_outputs, subtask_names, extra_keys):
    fc_keys = set().union(*(fc_output.keys() for fc_output in fc_outputs)) - {"idx"}
    collated_fc = {key: default_collate([fc_output[key] for fc_output in fc_outputs]) for key in fc_keys}
    if "idx" in fc_outputs[0]:
        collated_fc["idx"] = default_collate([fc_output["idx"] for fc_output in fc_outputs])

    allowed_input_keys = FC_KEYS | extra_keys | {"idx"}
    return {
        key: {subtask_name: value for subtask_name in subtask_names}
        for key, value in collated_fc.items()
        if key in allowed_input_keys
    }


def export_partition_pooled_embeddings(trainer, canonical_subtask_name, batch_size, obsm_key=None):
    data = trainer.data
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.eval()

    if obsm_key is None:
        obsm_key = f"{canonical_subtask_name}_pooled_embeddings"

    num_cells = data.adata.n_obs
    d_model = model.encoder.d_encoded
    if obsm_key not in data.adata.obsm:
        data.adata.obsm[obsm_key] = np.zeros((num_cells, d_model), dtype=np.float32)

    partition_label = "full_dataset"
    if getattr(data, "partition", None) is not None and hasattr(data, "partition_file_paths"):
        partition_index = data.partition
        partition_filename = Path(data.partition_file_paths[partition_index]).name
        partition_label = f"partition_idx={partition_index}, file={partition_filename}"

    cell_indices = np.arange(num_cells)
    subtask_names = [subtask_name for subtask_name, _ in trainer.data.tasklist]
    progress = tqdm(
        range(0, num_cells, batch_size),
        desc=f"Exporting pooled embeddings ({canonical_subtask_name}, {partition_label})",
        disable=not trainer.accelerator.is_main_process,
    )
    for start in progress:
        end = min(start + batch_size, num_cells)
        batch_indices = cell_indices[start:end]
        fc_outputs = [data.fc[int(idx)] for idx in batch_indices]
        cell_inputs = collate_fc_outputs(fc_outputs, subtask_names, extra_keys=data.fc.extra_keys)
        cell_inputs = {
            key: {
                subtask_name: value.to(trainer.accelerator.device) if value is not None else None
                for subtask_name, value in inner_dict.items()
            }
            for key, inner_dict in cell_inputs.items()
        }
        attention_mask = cell_inputs.get("expression_padding", {}).get(canonical_subtask_name)

        with torch.no_grad(), trainer.accelerator.autocast():
            sequence_embeddings = model.encode_cell(cell_inputs)[canonical_subtask_name]
            pooled_embeddings = pool_sequence_tokens(trainer, sequence_embeddings, attention_mask)

        data.adata.obsm[obsm_key][batch_indices] = pooled_embeddings.detach().cpu().float().numpy()


def export_pooled_embeddings(trainer, canonical_subtask_name=None, batch_size=None, obsm_key=None):
    if canonical_subtask_name is None:
        canonical_subtask_name, _ = next(iter(trainer.data.tasklist))
    if batch_size is None:
        batch_size = trainer.batchsize

    data = trainer.data

    # Partitioned CellRepresentation setup leaves a dummy partition open on each rank.
    # Drop it before enabling precomputed writes, otherwise every rank may try to rewrite
    # the same cached backed file during the first partition switch.
    if getattr(data, "adata", None) is not None and hasattr(data, "prepare_partition"):
        data.close_partition(is_original_replica=False)
        data._partition = None

    data._save_precomputed = True

    if hasattr(data, "num_partitions"):
        assigned_partitions = list(
            range(trainer.accelerator.process_index, data.num_partitions, trainer.accelerator.num_processes),
        )
        if trainer.accelerator.is_main_process:
            print(
                f"> Export helper assigned partitions for rank {trainer.accelerator.process_index}: {assigned_partitions}",
                flush=True,
            )
        for partition in assigned_partitions:
            data.partition = partition
            if (
                trainer.accelerator.is_main_process
                and getattr(data, "adata", None) is not None
                and getattr(data.adata, "filename", None) is not None
            ):
                active_path = Path(data.adata.filename).resolve()
                print(
                    f"> Export helper writing cached partition file: {active_path}",
                    flush=True,
                )
            export_partition_pooled_embeddings(
                trainer,
                canonical_subtask_name=canonical_subtask_name,
                batch_size=batch_size,
                obsm_key=obsm_key,
            )
        data.close_partition()
        data.partition = None
        trainer.accelerator.wait_for_everyone()
    else:
        export_partition_pooled_embeddings(
            trainer,
            canonical_subtask_name=canonical_subtask_name,
            batch_size=batch_size,
            obsm_key=obsm_key,
        )
