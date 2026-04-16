import pickle as pkl
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import default_collate

if TYPE_CHECKING:
    from Heimdall.cell_representations import CellRepresentation

from Heimdall.utils import clear_fully_qualified_cache_paths, get_fully_qualified_cache_paths, instantiate_from_config

CellFeatType = NDArray[np.int_] | NDArray[np.float32]
FeatType = CellFeatType | tuple[CellFeatType, CellFeatType]
LabelType = NDArray[np.int_] | NDArray[np.float32]


@dataclass
class Task(ABC):
    """Heimdall task key-value store.

    Contains information about an scFM task and training details. (Pre)computes
    labels for each task.

    """

    data: "CellRepresentation"
    task_type: str
    metrics: list[str]
    shuffle: bool
    # batchsize: int
    # epochs: int
    head_config: DictConfig
    loss_config: DictConfig
    interaction_type: str | None = None
    top_k: list[int] | None = None
    label_obsm_name: str | None = None
    label_col_name: str | None = None
    reducer_config: DictConfig | None = None
    splits: DictConfig | None = None
    train_split: float | None = (None,)
    track_metric: str | None = None
    # early_stopping: bool = False
    # early_stopping_patience: int = 5

    @property
    def labels(self) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        return getattr(self, "_labels", None)

    @labels.setter
    def labels(self, val) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        self._labels = val

    @property
    def num_tasks(self) -> int:
        if "_num_tasks" not in self.__dict__:
            # warnings.warn(
            #     "Need to improve to explicitly handle multiclass vs. multilabel",
            #     UserWarning,
            #     stacklevel=2,
            # )
            assert self.task_type in [
                "regression",
                "binary",
                "multiclass",
                "mlm",
            ], "task type must be regression, binary, multiclass or mlm. Check the task config file."

            task_type = self.task_type
            if task_type == "regression":
                if len(self.labels.shape) == 1:
                    out = 1
                else:
                    out = self._labels.shape[1]
            elif task_type == "binary":
                if len(self.labels.shape) == 1:
                    out = 1
                else:
                    out = self._labels.shape[1]
            elif task_type == "multiclass":
                out = self._labels.max() + 1
            elif task_type == "mlm":
                # out = self._labels.max() + 1
                out = self.labels.shape[0] + 1  # TODO why +1 ?
            else:
                raise ValueError(
                    f"Unknown task type {task_type!r}. Valid options are: 'multiclass', 'binary', 'regression', 'mlm'.",
                )

            self._num_tasks = out = int(out)
            print(
                f"> Task dimension: {out} " f"(task type {self.task_type!r}, {self.labels.shape=})",
            )

        return self._num_tasks

    @property
    def idx(self) -> NDArray[np.int_]:
        return self.data._idx

    @abstractmethod
    def setup_labels(self): ...

    def get_cache_path(self, cache_dir, hash_vars, task_name):
        processed_data_path, _, _ = get_fully_qualified_cache_paths(
            self.data.local_cfg,
            cache_dir,
            filename=f"{task_name}_labels.pkl",
            keys=(
                *self.data.TOKENIZER_KEYS,
                f"tasks.{task_name}",
            ),
            hash_vars=hash_vars,
        )
        return processed_data_path

    def clear_cache_path(self, cache_dir, hash_vars, task_name):
        return clear_fully_qualified_cache_paths(
            self.data.local_cfg,
            cache_dir,
            keys=(
                *self.data.TOKENIZER_KEYS,
                f"tasks.{task_name}",
            ),
            hash_vars=hash_vars,
        )

    def to_cache(self, cache_dir, hash_vars, task_name):
        processed_data_path = self.get_cache_path(cache_dir, hash_vars, task_name)
        with open(processed_data_path, "wb") as label_file:
            pkl.dump(self.labels, label_file)

        self.data.print_during_setup(f"> Finished writing task {task_name} labels at {processed_data_path}")

    def from_cache(self, cache_dir, hash_vars, task_name):
        processed_data_path = self.get_cache_path(cache_dir, hash_vars, task_name)
        if processed_data_path.is_file():
            self.data.print_during_setup(
                f"> Found already processed labels for task {task_name}: {processed_data_path}",
            )
            try:
                with open(processed_data_path, "rb") as label_file:
                    cached_labels = pkl.load(label_file)
            except (pkl.UnpicklingError, EOFError, ValueError) as exc:
                self.data.print_during_setup(
                    f"> Ignoring corrupted cached labels for task {task_name}: {processed_data_path} ({exc})",
                )
                return False

            if not self.validate_cached_labels(cached_labels):
                self.data.print_during_setup(
                    f"> Ignoring incompatible cached labels for task {task_name}: {processed_data_path}",
                )
                return False

            self.labels = cached_labels
            return True

        return False

    def validate_cached_labels(self, labels) -> bool:
        return True

    def get_inputs(self, idx, shared_inputs):
        return {
            "labels": self.labels[idx],
        }

    def on_batch(self):
        """Callback to reset task state on start of sampling batch."""
        return None

    def collate(self, values: list[Tensor | None]):
        # Drop Nones, or replace with zeros
        is_invalid = [v is None for v in values]
        if all(is_invalid):
            return None
        elif any(is_invalid):
            raise ValueError("Cannot have multiple samples with inhomogenous input validities.")
        else:
            collated_values = default_collate(values)

        return collated_values


class SingleInstanceTask(Task):
    def setup_labels(self):
        adata = self.data.adata
        if self.label_col_name is not None:
            assert self.label_obsm_name is None
            df = adata.obs
            class_mapping = {
                label: idx
                for idx, label in enumerate(
                    df[self.label_col_name].unique(),
                    start=0,
                )
            }
            df["class_id"] = df[self.label_col_name].map(class_mapping)
            labels = np.array(df["class_id"])
            if self.task_type == "regression":
                labels = labels.reshape(-1, 1).astype(np.float32)

        elif self.label_obsm_name is not None:
            assert self.label_col_name is None
            df = adata.obsm[self.label_obsm_name]

            if self.task_type == "binary":
                (labels := np.empty(df.shape, dtype=np.float32)).fill(np.nan)
                labels[np.where(df == 1)] = 1
                labels[np.where(df == -1)] = 0
            elif self.task_type == "regression":
                labels = np.array(df).astype(np.float32)

            print(f"labels shape {labels.shape}")

        else:
            raise ValueError("Either 'label_col_name' or 'label_obsm_name' needs to be set.")

        self.labels = labels


class DummyTask(Task):
    def setup_labels(self):
        self.labels = np.zeros((len(self.data.adata), 1), dtype=np.float32)


class AutoencoderTask(Task):
    """Reconstruct the current batch expression inputs with an MSE loss.

    This is intended for use with `model=expression_only`, where the model
    forwards `expression_inputs` directly and the head learns a reconstruction
    of that expression vector.

    """

    @property
    def num_tasks(self) -> int:
        return len(self.data.tokenizer_context.raw_gene_names)

    def setup_labels(self):
        # Keep cached task metadata small; per-batch labels are taken directly
        # from shared tokenizer outputs in `get_inputs`.
        self.labels = np.empty((len(self.data.adata), 1), dtype=np.float32)

    def validate_cached_labels(self, labels) -> bool:
        return isinstance(labels, np.ndarray) and labels.shape == (len(self.data.adata), 1)

    def get_inputs(self, idx, shared_inputs):
        return {
            "labels": np.asarray(shared_inputs["expression_inputs"], dtype=np.float32),
        }


class PairedInstanceTask(Task):
    def setup_labels(self):
        adata = self.data.adata
        full_mask = adata.obsp["full_mask"]
        nz = np.nonzero(full_mask)

        # Task type specific handling
        task_type = self.task_type
        if task_type == "multiclass":
            if len(self.data.obsp_task_keys) > 1:
                raise ValueError(
                    f"{task_type!r} only supports a single task key, provided task keys: {self.data.obsp_task_keys}",
                )

            task_mat = adata.obsp[self.data.obsp_task_keys[0]]
            num_tasks = task_mat.max()  # class id starts from 1. 0's are ignoreed
            labels = np.array(task_mat[nz]).ravel().astype(np.int64) - 1  # class 0 is not used

        elif task_type == "binary":
            num_tasks = len(self.data.obsp_task_keys)

            (labels := np.empty((len(nz[0]), num_tasks), dtype=np.float32)).fill(np.nan)
            for i, task in enumerate(self.data.obsp_task_keys):
                label_i = np.array(adata.obsp[task][nz]).ravel()
                labels[:, i][label_i == 1] = 1
                labels[:, i][label_i == -1] = 0

        elif task_type == "regression":
            num_tasks = len(self.data.obsp_task_keys)

            labels = np.zeros((len(nz[0]), num_tasks), dtype=np.float32)
            for i, task in enumerate(self.data.obsp_task_keys):
                labels[:, i] = np.array(adata.obsp[task][nz]).ravel()

        else:
            raise ValueError(f"task_type must be one of: 'multiclass', 'binary', 'regression'. Got: {task_type!r}")

        self.labels = labels


class MLMMixin:
    def get_inputs(self, idx, shared_inputs):
        identity_inputs = shared_inputs["identity_inputs"]
        return {
            "identity_inputs": identity_inputs,
            "labels": identity_inputs.astype(int),
        }

    def setup_labels(self):
        # Dummy labels to indicate task size
        self.labels = np.empty(self.data.fg.vocab_size)


class MaskedMixin(ABC):
    def __init__(self, *args, mask_ratio: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio

    @property
    @abstractmethod
    def mask_token(self): ...


class TransformationMixin(ABC):
    def get_inputs(self, idx, shared_inputs):
        data = super().get_inputs(idx, shared_inputs)
        return self._transform(data)

    @abstractmethod
    def _transform(self, data): ...


class SeqMaskedMLMTask(TransformationMixin, MaskedMixin, MLMMixin, SingleInstanceTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._num_tasks = self.data.adata.n_vars  # number of genes

    @property
    def mask_token(self):
        return self.data.special_tokens["mask"]

    def _transform(self, data):
        size = data["labels"].size
        mask = np.random.random(size) < self.mask_ratio

        # Ignore padding tokens
        is_padding = data["labels"] == self.data.special_tokens["pad"]
        mask[is_padding] = False

        negative_mask = data["identity_inputs"] < 0
        mask = (mask * ~negative_mask).astype(bool)

        data["identity_inputs"][mask] = self.mask_token

        data["identity_inputs"][mask] = self.mask_token
        # data["expression_inputs"][mask] = self.mask_token
        data["masks"] = mask

        return data


@dataclass
class ContrastiveViewTask(Task):
    min_panel_size: int = 400
    batchsize: int | None = None

    @property
    def rng(self):
        return self.data.rng

    @property
    def num_raw_genes(self):
        return len(self.data.raw_gene_names)

    def log_sample(self, min_genes, max_genes):
        if min_genes >= max_genes:
            return min_genes

        log_sample = self.rng.uniform(np.log2(min_genes), np.log2(max_genes))
        sample = int(np.exp2(log_sample))

        return max(min(sample, max_genes), min_genes)

    def setup_labels(self):
        # Dummy labels used to size the contrastive head to the training batch.
        if self.batchsize is None:
            raise ValueError("`ContrastiveViewTask` requires `args.batchsize` to be set in the task config.")
        batchsize = self.batchsize
        self.labels = np.zeros((len(self.data.adata), batchsize), dtype=np.float32)

    def validate_cached_labels(self, labels) -> bool:
        if self.batchsize is None:
            return False
        batchsize = self.batchsize
        expected_shape = (len(self.data.adata), batchsize)
        return isinstance(labels, np.ndarray) and labels.shape == expected_shape

    def get_inputs(self, idx, shared_inputs):
        if not (hasattr(self, "panel_1_idx") and hasattr(self, "panel_2_idx")):
            self.on_batch()

        labels = self.labels[shared_inputs["idx"]]

        self.data.fe.gene_panel_idx = self.panel_1_idx
        inputs_1 = self.data.tokenizer[idx]

        self.data.fe.gene_panel_idx = self.panel_2_idx
        inputs_2 = self.data.tokenizer[idx]

        # Reset back to the full gene panel after building the two views.
        self.data.fe.gene_panel_idx = np.arange(self.data.num_genes)

        view_inputs = [inputs_1, inputs_2]
        inputs = {key: [view_input[key] for view_input in view_inputs] for key in inputs_1}
        inputs["labels"] = [labels for _ in view_inputs]

        return inputs

    def on_batch(self):
        """Sample two independent, non-overlapping gene panels for this
        batch."""
        all_indices = np.arange(self.num_raw_genes)
        panel_size_1 = self.log_sample(self.min_panel_size, self.num_raw_genes)
        panel_idx_1 = self.rng.choice(all_indices, panel_size_1, replace=False)

        remaining_indices = np.setdiff1d(all_indices, panel_idx_1, assume_unique=True)
        if len(remaining_indices) < self.min_panel_size:
            panel_idx_2 = remaining_indices
        else:
            panel_size_2 = self.log_sample(self.min_panel_size, self.num_raw_genes - panel_size_1)
            panel_idx_2 = self.rng.choice(remaining_indices, panel_size_2, replace=False)
            assert np.intersect1d(panel_idx_1, panel_idx_2).size == 0, "Panels overlap"

        self.panel_1_idx = panel_idx_1
        self.panel_2_idx = panel_idx_2

    def collate(self, values: list[Tensor | None]):
        is_invalid = [v is None for v in values]
        if all(is_invalid):
            return None
        elif any(is_invalid):
            raise ValueError("Cannot have multiple samples with inhomogenous input validities.")

        first_value = values[0]
        if not isinstance(first_value, (list, tuple)):
            return default_collate(values)

        view_1_values = [view_1 for view_1, _ in values]
        view_2_values = [view_2 for _, view_2 in values]
        return default_collate(view_1_values + view_2_values)


class Tasklist:
    """Container for multiple Heimdall tasks.

    Tasks must use the same `Dataset` object config and splits/dataloader.

    """

    PROPERTIES = (
        "splits",
        "interaction_type",
        "shuffle",
        # "batchsize",
        # "epochs",
        # "early_stopping",
        # "early_stopping_patience",
    )

    def __init__(
        self,
        data: "CellRepresentation",
        tasks: DictConfig | dict,
    ):
        if not tasks:
            raise ValueError("Tasklist requires at least one task configuration.")
        self.data = data
        self._tasks = {
            subtask_name: instantiate_from_config(subtask_config, data)
            for subtask_name, subtask_config in tasks.items()
        }

        self.set_unique_properties()
        self.num_subtasks = len(self._tasks)

    def set_unique_properties(self):
        for property_name in self.PROPERTIES:
            unique_properties = {getattr(task, property_name, None) for task in self._tasks.values()}
            if len(unique_properties) > 1:
                raise ValueError(f"All tasks must use the same `{property_name}` value.")

            unique_property = next(iter(unique_properties))
            setattr(self, property_name, unique_property)

    def __getitem__(self, key: str | None):
        if key is None:
            if len(self._tasks) > 1:
                raise ValueError("`None` key only works if `TaskList` contains a singular item.")

            return next(iter(self._tasks.values()))

        return self._tasks[key]

    def __setitem__(self, key: str, value: Task):
        self._tasks[key] = value
        self.num_subtasks = len(self._tasks)

    def __delitem__(self, key: str):
        del self._tasks[key]
        self.num_subtasks = len(self._tasks)

    def __iter__(self):
        yield from self._tasks.items()

    @property
    def active_items(self):
        return tuple(
            (subtask_name, subtask)
            for subtask_name, subtask in self._tasks.items()
            if not getattr(subtask, "is_pseudo", False)
        )

    def __len__(self):
        return self.num_subtasks
