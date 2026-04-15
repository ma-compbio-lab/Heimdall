"""The Cell Representation Object for Processing."""

import pickle as pkl
import textwrap
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from pprint import pformat
from typing import Dict, Union

import anndata as ad
import numpy as np
import scanpy as sc
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from numpy.typing import NDArray
from omegaconf import OmegaConf, open_dict
from scipy import sparse
from scipy.sparse import csc_array, csr_array
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from Heimdall.datasets import Dataset, PartitionedSubset
from Heimdall.fc import Fc
from Heimdall.fe import Fe
from Heimdall.fg import Fg
from Heimdall.samplers import DefaultBatchSampler, PartitionedBatchSampler, PartitionedDistributedSampler
from Heimdall.task import Tasklist
from Heimdall.tokenizer import Tokenizer

# from Heimdall.samplers import PartitionedDistributedSampler
from Heimdall.utils import (
    check_states,
    clear_fully_qualified_cache_paths,
    conditional_print,
    convert_to_ensembl_ids,
    get_collation_closure,
    get_fully_qualified_cache_paths,
    instantiate_from_config,
    issparse,
)


class SpecialTokenMixin:
    _SPECIAL_TOKENS = ["pad", "mask"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.special_tokens = {token: self.num_genes + i for i, token in enumerate(self._SPECIAL_TOKENS)}


class CellRepresentation(SpecialTokenMixin):
    DATASET_KEYS = ("dataset.preprocess_args.data_path", "tasks")
    TOKENIZER_KEYS = ("dataset.preprocess_args.data_path", "tokenizer_context", "fg", "fe", "fc")

    @property
    def local_cfg(self):
        return self._cfg.scfm

    @property
    def fg_cfg(self):
        return self._cfg.scfm.fg

    @property
    def fe_cfg(self):
        return self._cfg.scfm.fe

    @property
    def fc_cfg(self):
        return self._cfg.scfm.fc

    def __init__(self, config, accelerator: Accelerator, auto_setup: bool = True, indent: int = 0):
        """Initialize the Cell Rep object with configuration and AnnData object.

        Parameters:
        config (dict): Configuration dictionary.

        """
        self.rank = accelerator.process_index
        self.num_replicas = accelerator.num_processes
        self.accelerator = accelerator
        self._indent = indent
        self._save_precomputed = False
        self._get_precomputed = False

        self.setup_finished = False
        self._cfg = config
        self.data_path = str(self.local_cfg.dataset.preprocess_args.data_path)

        self.float_dtype = config.float_dtype
        self.adata = None
        self.processed_fcfg = False
        self.verbose = 0  # TODO: expose

        seed = 0  # TODO: make this configurable???
        self.rng = np.random.default_rng(seed)

        if auto_setup:
            if config.overwrite:
                if self.accelerator.is_main_process:
                    self.clear_cache()
                self.accelerator.wait_for_everyone()
            self.create_tasklist()
            self.setup(ops=("preprocess",))
            SpecialTokenMixin.__init__(self)
            self.prepare_full_dataset()
            self.setup(ops=("labels",))
            self.setup_finished = True
            self.prepare_dataset_loaders()

    @property
    def indent(self):
        return self._indent

    @indent.setter
    def indent(self, val: int):
        self._indent = val

    @property
    def prefix(self):
        return " " * (self.indent * 4)

    @property
    def save_precomputed(self):
        return self._save_precomputed

    @property
    def get_precomputed(self):
        return self._get_precomputed

    def setup_labels(self, hash_vars=()):
        """Can only be called after `self.adata` and `self.datasets` is
        populated."""

        if not hasattr(self, "datasets"):
            return

        for subtask_name, subtask in self.tasklist:
            if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
                cache_dir = Path(cache_dir)
                is_cached = subtask.from_cache(
                    cache_dir / "processed_data",
                    hash_vars=hash_vars,
                    task_name=subtask_name,
                )
                if is_cached:
                    continue

            subtask.setup_labels()
            if cache_dir is not None:
                subtask.to_cache(cache_dir / "processed_data", hash_vars=hash_vars, task_name=subtask_name)

        self.print_during_setup("> Finished setting up labels")

    def setup(self, hash_vars=(), ops=("preprocess", "labels")):
        if "preprocess" in ops:
            self.load_anndata(hash_vars=hash_vars)
            self.setup_tokenizer(hash_vars=hash_vars)
        if "labels" in ops:
            self.setup_labels(hash_vars=hash_vars)
        # if hasattr(self, "datasets") and "full" in self.datasets:
        #     self.prepare_dataset_loaders()

    @property
    @check_states(labels=True)
    def labels(self) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        labels = {subtask_name: subtask.labels for subtask_name, subtask in self.tasklist}
        return labels

    @labels.setter
    def labels(self, val) -> Union[NDArray[np.int_], NDArray[np.float32]]:
        for subtask_name, subtask in self.tasklist:
            subtask.labels = val[subtask_name]

    @property
    @check_states(splits=True)
    def splits(self) -> Dict[str, NDArray[np.int_]]:
        return self._splits

    @property
    def raw_gene_names(self):
        return self._raw_gene_names

    @raw_gene_names.setter
    def raw_gene_names(self, val):
        if hasattr(self, "_raw_gene_names") and (val != self._raw_gene_names).any():
            raise ValueError("Raw gene names must match for all partitions.")

        self._raw_gene_names = val

    @property
    def gene_names(self, mask_key: str = "identity_valid_mask"):
        if self.fg.identity_valid_mask is not None:
            valid_mask = np.asarray(self.fg.identity_valid_mask, dtype=bool)
            return self.raw_gene_names[valid_mask]
        if mask_key in self.adata.var:
            valid_mask = self.adata.var[mask_key].to_numpy(dtype=bool, copy=False)
            return self.raw_gene_names[valid_mask]
        return self.raw_gene_names

    @property
    def num_genes(self):
        return len(self.gene_names)

    def convert_to_ensembl_ids(self, data_dir, species="human"):
        """Converts gene symbols in the anndata object to Ensembl IDs using a
        provided mapping.

        Args:
            - data: anndata object with gene symbols as var index
            - data_dir: directory where the data is stored
            - species: species name (default is "human")

        Returns:
            - data: anndata object with Ensembl IDs as var index
            - symbol_to_ensembl_mapping: mapping dictionary from symbols to Ensembl IDs

        """

        _, gene_mapping = convert_to_ensembl_ids(
            self.adata,
            data_dir,
            species=species,
            verbose=self.verbose and not self.setup_finished,
        )
        return self.adata, gene_mapping

    def clear_cache(self, hash_vars=()):
        """Empty cache."""
        cache_dir = self._cfg.cache_preprocessed_dataset_dir
        if cache_dir is None:
            return

        cache_dir = Path(cache_dir)

        # Clear preprocessed dataset
        cleared_processed_dir = clear_fully_qualified_cache_paths(
            self.local_cfg,
            cache_dir / "processed_anndata",
            keys=self.DATASET_KEYS,
        )
        self.print_during_setup(f"> Cleared `processed_anndata` cache at {cleared_processed_dir}.")

        # Clear tokenizer
        cleared_tokenizer_dir = clear_fully_qualified_cache_paths(
            self.local_cfg,
            cache_dir / "processed_data",
            keys=self.TOKENIZER_KEYS,
            hash_vars=hash_vars,
        )
        self.print_during_setup(f"> Cleared tokenizer cache at {cleared_tokenizer_dir}.")

        # Clear labels
        cleared_label_dirs = []
        for subtask_name, subtask in self.tasklist:
            cleared_label_dirs.append(
                subtask.clear_cache_path(cache_dir / "processed_data", hash_vars=hash_vars, task_name=subtask_name),
            )

        self.print_during_setup(f"> Cleared label cache at {cleared_label_dirs}.")

    def anndata_from_cache(self, preprocessed_data_path, preprocessed_cfg_path, cfg):
        if preprocessed_data_path.is_file():
            self.print_during_setup(
                f"> Found already preprocessed anndata: {preprocessed_data_path}",
            )
            self.adata = ad.read_h5ad(
                preprocessed_data_path,
                backed="r",
            )  # add backed argument to prevent entire dataset from being read into mem
            self.raw_gene_names = self.adata.var_names.copy()
            return True

        # OmegaConf.save(cfg, preprocessed_cfg_path)

        return False

    def anndata_to_cache(self, preprocessed_data_path):
        if preprocessed_data_path is not None:
            self.print_during_setup(
                f"> Writing preprocessed Anndata Object at {preprocessed_data_path}",
                is_printable_process=True,
            )
            self.adata.write(preprocessed_data_path)
            self.print_during_setup(
                f"> Finished writing preprocessed Anndata Object at {preprocessed_data_path}",
                is_printable_process=True,
            )

    def create_tasklist(self):
        tasks = self.local_cfg.tasks
        if not tasks:
            raise ValueError("No tasks configured. Add at least one task under `*.tasks.<task_name>`.")

        self.tasklist = Tasklist(self, tasks=tasks)

        self.num_subtasks = self.tasklist.num_subtasks

    def load_anndata(self, filename: str = "data.h5ad", hash_vars=()):
        """Load AnnData into memory (and preprocess, if necessary)."""
        if self.adata is not None:
            raise ValueError("Anndata object already exists, are you sure you want to reprocess again?")

        preprocessed_data_path = preprocessed_cfg_path = cfg = None
        if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
            cache_dir = Path(cache_dir)
            preprocessed_data_path, preprocessed_cfg_path, cfg = get_fully_qualified_cache_paths(
                self.local_cfg,
                cache_dir / "processed_anndata",
                filename,
                keys=self.DATASET_KEYS,
                hash_vars=hash_vars,
            )
            is_cached = self.anndata_from_cache(preprocessed_data_path, preprocessed_cfg_path, cfg)

            if is_cached:
                self.print_during_setup(f"> Finished loading AnnData with shape: {self.adata.shape}")
                return

        self.preprocess_anndata()
        self.anndata_to_cache(preprocessed_data_path)

        self.print_during_setup(f"> Finished loading AnnData with shape: {self.adata.shape}")

    def preprocess_anndata(self):
        self.adata = ad.read_h5ad(self.data_path)

        self.print_during_setup(f"> Finished Loading in {self.local_cfg.dataset.preprocess_args.data_path}")
        # convert gene names to ensembl ids
        self.print_during_setup("> Converting gene names to Ensembl IDs...")
        self.adata, _ = self.convert_to_ensembl_ids(
            data_dir=self._cfg.ensembl_dir,
            species=self.local_cfg.dataset.preprocess_args.species,
        )

        # for key in ('gene_ensembl',):
        #     if key in self.adata.var:
        #         del self.adata.var[key]

        for key in ("gene_mapping:ensembl_to_symbol", "gene_mapping:symbol_to_ensembl"):
            if key in self.adata.uns:
                del self.adata.uns[key]

        if self.local_cfg.dataset.preprocess_args.get("normalize", False):
            self.print_during_setup("> Normalizing AnnData...")

            if sparse.issparse(self.adata.X):
                data = self.adata.X.data
            else:
                data = self.adata.X

            # Store mask of NaNs
            nan_mask = np.isnan(data)

            if np.any(nan_mask):
                data[nan_mask] = 0
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            if np.any(nan_mask):
                data[nan_mask] = np.nan  # NOTE: must not be integer-valued

            assert (
                self.local_cfg.dataset.preprocess_args.normalize and self.local_cfg.dataset.preprocess_args.log_1p
            ), "Normalize and Log1P both need to be TRUE"
        else:
            self.print_during_setup("> Skipping Normalizing anndata...")

        if self.local_cfg.dataset.preprocess_args.get("log_1p", False):
            self.print_during_setup("> Log Transforming anndata...")

            sc.pp.log1p(self.adata)
        else:
            self.print_during_setup("> Skipping Log Transforming anndata..")

        if (
            self.local_cfg.dataset.preprocess_args.get("top_n_genes", False)
            and self.local_cfg.dataset.preprocess_args["top_n_genes"] != "false"
        ):
            # Identify highly variable genes
            self.print_during_setup(
                f"> Using highly variable subset... top {self.local_cfg.dataset.preprocess_args.top_n_genes} genes",
            )
            sc.pp.highly_variable_genes(self.adata, n_top_genes=self.local_cfg.dataset.preprocess_args.top_n_genes)
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()
        else:
            self.print_during_setup("> No highly variable subset... using entire dataset")

        if self.local_cfg.dataset.preprocess_args.get("scale_data", False):
            # Scale the data
            raise NotImplementedError("Scaling the data is NOT RECOMMENDED, please set it to false")
            self.print_during_setup("> Scaling the data...")
            sc.pp.scale(self.adata, max_value=10)
        else:
            self.print_during_setup("> Not scaling the data...")

        if self.local_cfg.dataset.preprocess_args.get("get_medians", False):
            # Get medians
            self.print_during_setup("> Getting nonzero medians...")
            csc_expression = csc_array(self.adata.X)
            genewise_nonzero_expression = np.split(csc_expression.data, csc_expression.indptr[1:-1])
            gene_medians = np.array([np.median(gene_nonzeros) for gene_nonzeros in genewise_nonzero_expression])
            self.adata.var["medians"] = gene_medians

        if not issparse(self.adata.X):
            if getattr(self.adata, "isbacked", False):
                # TODO: add back with verbose
                pass
                # print("> Data is dense and backed, skipping conversion to CSR to keep memory mapping.")
            else:
                # print("> Data was provided in dense format, converting to CSR. Consider precomputing.")
                self.adata.X = csr_array(self.adata.X)

        self.raw_gene_names = self.adata.var_names.copy()
        self.print_during_setup(f"> Finished processing AnnData object:\n{self.adata}")

    @check_states(processed_fcfg=True)
    def prepare_full_dataset(self):
        # Set up full dataset given the processed cell representation data
        # This will prepare: labels, splits
        full_dataset: Dataset = instantiate_from_config(self.local_cfg.trainer.dataset_config, self)
        self.datasets = {"full": full_dataset}

    @check_states(adata=True, processed_fcfg=True)
    def prepare_dataset_loaders(self):
        full_dataset = self.datasets["full"]
        # Set up dataset splits given the data splits
        for split, split_idx in self.splits.items():
            self.datasets[split] = Subset(full_dataset, split_idx)

        # Set up data loaders
        # dataloader_kwargs = {}  # TODO: USE THIS IF DEBUGGING
        heimdall_collate_fn = get_collation_closure(self.tasklist)
        dataloader_kwargs = {"num_workers": 4}  # TODO: we can parse additional data loader kwargs from config
        per_device_batch_size = self.local_cfg.trainer.args.batchsize // self.accelerator.num_processes
        self.dataloaders = {
            split: DataLoader(
                dataset,
                batch_sampler=DefaultBatchSampler(
                    sampler=(
                        RandomSampler(dataset)
                        if split == "train" and self.tasklist.shuffle
                        else SequentialSampler(dataset)
                    ),
                    batch_size=per_device_batch_size,
                    drop_last=False,
                    tasklist=self.tasklist,
                ),
                collate_fn=heimdall_collate_fn,
                **dataloader_kwargs,
            )
            for split, dataset in self.datasets.items()
        }

        dataset_str = pformat(self.datasets).replace("\n", "\n\t")
        self.print_during_setup(
            f"> Finished setting up datasets (and loaders):\n\t{dataset_str}",
            is_printable_process=True,
        )

    def get_tokenizer_cache_path(self, cache_dir, hash_vars, filename: str = "data.pkl"):
        processed_data_path, _, _ = get_fully_qualified_cache_paths(
            self.local_cfg,
            cache_dir / "processed_data",
            filename,
            keys=self.TOKENIZER_KEYS,
            hash_vars=hash_vars,
        )

        return processed_data_path

    def load_tokenizer_from_cache(self, cache_dir, hash_vars):
        processed_data_path = self.get_tokenizer_cache_path(cache_dir, hash_vars)
        if processed_data_path.is_file():
            self.print_during_setup(
                f"> Found already processed `CellRepresentation`: {processed_data_path}",
            )
            # loaded_cfg_str = OmegaConf.to_yaml(OmegaConf.load(processed_cfg_path)).replace("\n", "\n    ")
            # print(f"  Processing config:\n    {loaded_cfg_str}") # TODO: add verbosity levels

            with open(processed_data_path, "rb") as rep_file:
                (
                    identity_embedding_index,
                    identity_valid_mask,
                    gene_embeddings,
                    expression_embeddings,
                ) = pkl.load(rep_file)

            self.tokenizer.load_from_cache(
                identity_embedding_index,
                identity_valid_mask,
                gene_embeddings,
                expression_embeddings,
            )

            self.processed_fcfg = True

            return True

        # TODO: add back
        # OmegaConf.save(cfg, processed_cfg_path)
        return False

    def save_tokenizer_to_cache(self, cache_dir, hash_vars):
        processed_data_path = self.get_tokenizer_cache_path(cache_dir, hash_vars)
        if not processed_data_path.is_file():
            with open(processed_data_path, "wb") as rep_file:
                cache_representation = self.tokenizer.get_cache_state()
                pkl.dump(cache_representation, rep_file)
                self.print_during_setup(f"> Finished writing cell representations at {processed_data_path}")

    def instantiate_tokenizer(self):
        """Instantiate tokenizer context and `Fg`, `Fe`, `Fc` according to
        config."""
        self.tokenizer = Tokenizer(
            self._cfg,
            adata=self.adata,
            raw_gene_names=self.raw_gene_names,
            rng=self.rng,
            float_dtype=self.float_dtype,
            verbose=self.verbose,
        )
        self.tokenizer_context = self.tokenizer.context
        self.fg: Fg = self.tokenizer.fg
        self.fe: Fe = self.tokenizer.fe
        self.fc: Fc = self.tokenizer.fc

    # @check_states(adata=True)
    def setup_tokenizer(self, hash_vars=()):
        """Processes the `f_g`, `fe` and `f_c` from the config.

        This will first check to see if the cell representations are already
        cached, and then will either load the cached representations or compute
        them and save them.

        """

        self.instantiate_tokenizer()
        if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
            cache_dir = Path(cache_dir)
            is_cached = self.load_tokenizer_from_cache(cache_dir, hash_vars=hash_vars)
            if is_cached:
                return

        self.tokenizer.preprocess_embeddings()
        self.print_during_setup(f"> Finished calculating fg with {self.fg_cfg.type}")
        self.print_during_setup(f"> Finished calculating fe with {self.fe_cfg.type}")

        self.processed_fcfg = True

        if cache_dir is not None:
            self.save_tokenizer_to_cache(cache_dir, hash_vars=hash_vars)

    def print_r0(self, message):
        conditional_print(f"{message}", self.accelerator.is_main_process)

    def print_during_setup(self, message, is_printable_process=False):
        message = textwrap.indent(message, self.prefix)
        # message = self.indent + message
        if not self.setup_finished:
            if is_printable_process:
                print(message)
            else:
                self.print_r0(message)


class PartitionedCellRepresentation(CellRepresentation):
    def __init__(self, config, accelerator: Accelerator, auto_setup: bool = True, indent: int = 0):
        super().__init__(config, accelerator, auto_setup=False, indent=indent)

        # Expect `self._cfg.dataset.preprocess_args.data_path` to hold parent directory, not filepath
        self.partition_folder = str(self.local_cfg.dataset.preprocess_args.data_path)
        self.partition_file_paths = sorted(
            Path(self.partition_folder).glob("*.h5ad"),
        )
        self.num_partitions = len(self.partition_file_paths)

        if self.num_partitions == 0:
            raise ValueError(
                "No partitions were found under the directory at "
                f"'{self.local_cfg.dataset.preprocess_args.data_path}'. The dataset path "
                "(`config.dataset.preprocess_args.data_path`) is probably set incorrectly.",
            )

        self.partition_sizes = {}
        self.num_cells = {}
        if auto_setup:
            self.create_tasklist()

            if (cache_dir := self._cfg.cache_preprocessed_dataset_dir) is not None:
                cache_dir = Path(cache_dir)
                metadata_path, _, _ = get_fully_qualified_cache_paths(
                    self.local_cfg,
                    cache_dir / "processed_anndata",
                    "metadata.pkl",
                    keys=self.DATASET_KEYS,
                    hash_vars=(),
                )

            if self._cfg.overwrite:
                if self.accelerator.is_main_process:
                    self.clear_cache()
                self.accelerator.wait_for_everyone()

            if cache_dir is not None and metadata_path.is_file():
                self.print_during_setup(f"> Found partition metadata at {metadata_path}")

                with open(metadata_path, "rb") as metadata_file:
                    (
                        self.partition_sizes,
                        self.num_cells,
                        self._raw_gene_names,
                    ) = pkl.load(metadata_file)

                # Using zeroth partition as dummy
                for rank in range(self.num_replicas):
                    if rank == self.rank:
                        self.print_during_setup(
                            f"> Rank {self.rank}: starting dummy partition loading...",
                            is_printable_process=True,
                        )
                        self.indent += 1
                        dummy_partition = 0
                        self.prepare_partition(dummy_partition)
                        super().setup(hash_vars=(int(dummy_partition),), ops=("preprocess",))

                        expected_partition_size = self.get_stale_metadata_expected_size(dummy_partition)
                        actual_partition_size = self.get_stale_metadata_actual_size()
                        size_name = self.get_stale_metadata_size_name()
                        if expected_partition_size is not None and int(expected_partition_size) != int(
                            actual_partition_size,
                        ):
                            raise RuntimeError(
                                "Cached partition metadata is stale: "
                                f"partition {dummy_partition} expected {size_name} {expected_partition_size}, "
                                f"but current data has {size_name} {actual_partition_size}. "
                                f"Remove stale metadata at {metadata_path} and rerun.",
                            )

                        self.prepare_full_dataset()
                        SpecialTokenMixin.__init__(self)
                        self.partition = None
                        self.indent -= 1
                        self.print_during_setup(
                            f"> Rank {self.rank}: finished dummy partition loading.",
                            is_printable_process=True,
                        )
                    self.print_during_setup(
                        f"> Rank {self.rank}: waiting at dummy partition barrier for loop rank {rank}.",
                        is_printable_process=True,
                    )
                    self.accelerator.wait_for_everyone()

            else:
                self.print_during_setup("> Setting up partition_sizes...")
                self.indent += 1
                self.setup(ops=("preprocess",))  # One time through for main process
                self.indent -= 1
                self.print_during_setup("> Finished setting up partition_sizes")

                self.prepare_full_dataset()  # Setup dataset before preparing labels

                self.partition = None

                self.print_during_setup("> Setting up labels...")
                self.indent += 1
                self.setup()
                self.indent -= 1
                self.print_during_setup("> Finished setting up labels")

                if cache_dir is not None:
                    metadata_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(metadata_path, "wb") as metadata_file:
                        metadata = (
                            self.partition_sizes,
                            self.num_cells,
                            self._raw_gene_names,
                        )
                        pkl.dump(metadata, metadata_file)
                    self.print_during_setup(f"> Saved partition metadata at {metadata_path}")

            self.setup_finished = True
            self.print_during_setup("> Finished setting up labels")

            self.prepare_dataset_loaders()
            self._initialize_partition()

    def _initialize_partition(self):
        self.dataloaders["full"].batch_sampler.sampler.partition_idx = 0

    def get_stale_metadata_expected_size(self, partition: int):
        return self.partition_sizes.get(partition)

    def get_stale_metadata_actual_size(self):
        return self.adata.n_obs

    def get_stale_metadata_size_name(self):
        return "size"

    def set_partition_size(self):
        """Get the size of the current partition."""
        self.partition_sizes[self.partition] = self.adata.n_obs
        self.num_cells[self.partition] = self.adata.n_obs

    def clear_cache(self):
        """Empty cache."""
        for partition in range(self.num_partitions):  # Setting up AnnData and sizes
            super().clear_cache(hash_vars=(int(partition),))

    def setup(self, ops=("preprocess", "labels")):
        for rank in range(self.num_replicas):
            if rank == self.rank:
                self.print_during_setup(
                    f"> Rank {self.rank}: starting partition setup pass {ops=}.",
                    is_printable_process=True,
                )
                self._setup_partitions(ops=ops)
                self.print_during_setup(
                    f"> Rank {self.rank}: finished partition setup pass {ops=}.",
                    is_printable_process=True,
                )
            self.print_during_setup(
                f"> Rank {self.rank}: waiting at partition setup barrier for loop rank {rank}.",
                is_printable_process=True,
            )
            self.accelerator.wait_for_everyone()

    def _setup_partitions(self, ops):
        for partition in range(self.num_partitions):  # Setting up AnnData and sizes
            self.prepare_partition(partition)

            self.indent += 1
            super().setup(hash_vars=(int(self.partition),), ops=ops)
            self.indent -= 1

            self.set_partition_size()

        SpecialTokenMixin.__init__(self)  # TODO: this works because all partitions have the
        # same `var_names`. Can we actually enforce that during preprocessing?

        self.partition = None

    def preprocess_anndata(self):
        self.data_path = str(self.partition_file_paths[self.partition])
        # print(f"[Rank {self.accelerator.process_index}] Setting to {self.data_path=}")
        super().preprocess_anndata()

    def load_anndata(self, filename="data.h5ad", hash_vars=()):
        partition_source_path = self.partition_file_paths[self.partition]
        partition_filename = Path(partition_source_path).name
        # Keep all partition AnnData caches in the same dataset-level cache directory.
        # The filenames are already unique per partition.
        super().load_anndata(partition_filename, hash_vars=())
        if getattr(self.adata, "filename", None) is None and self._cfg.cache_preprocessed_dataset_dir is not None:
            cache_dir = Path(self._cfg.cache_preprocessed_dataset_dir)
            preprocessed_data_path, _, _ = get_fully_qualified_cache_paths(
                self.local_cfg,
                cache_dir / "processed_anndata",
                partition_filename,
                keys=self.DATASET_KEYS,
                hash_vars=(),
            )
            if preprocessed_data_path.is_file():
                self.adata = ad.read_h5ad(
                    preprocessed_data_path,
                    backed="r",
                )
        print(
            f"> Loaded partition {self.partition + 1} cache from "
            f"{getattr(self.adata, 'filename', None)!r} with obsm_keys={sorted(self.adata.obsm.keys())}",
        )

    def close_partition(self, is_original_replica: bool = True):
        """Close current partition."""

        if self.adata is not None:
            self.print_during_setup(
                f"> Closing partition {self.partition + 1} of {self.num_partitions}",
            )
            if self.save_precomputed and is_original_replica:
                # print(f"[Rank {self.accelerator.process_index}] Writing {self.adata.filename=}")
                filename = Path(self.adata.filename)
                adata_copy = self.adata.to_memory()
                self.adata.file.close()
                adata_copy.write_h5ad(filename)
                # print(f'[Rank {self.accelerator.process_index}] Wrote {self.adata.obsm.keys()=}')
            else:
                self.adata.file.close()

            del self.adata
            self.adata = None

    @property
    def partition(self):
        return self._partition

    def prepare_partition(self, partition):
        self.close_partition()
        self._partition = partition
        if partition is not None:
            self.print_during_setup(
                f"> Opening partition {partition + 1} of {self.num_partitions}",
            )
            return True

        return False

    @partition.setter
    def partition(self, partition):
        """Move to a new partition."""
        if getattr(self, "_partition", None) == partition:
            return

        # Preprocess partition AnnData
        partition_prepared = self.prepare_partition(partition)
        if partition_prepared:
            super().setup(
                hash_vars=(int(self.partition),),
                ops=("preprocess", "labels"),
            )

    @check_states(processed_fcfg=True)
    def prepare_dataset_loaders(self):

        # Set up dataset splits given the data splits
        overall_splits = defaultdict(lambda: defaultdict(dict))
        full_dataset = self.datasets["full"]
        for partition, splits in full_dataset.partition_splits.items():
            for split, split_idx in splits.items():
                overall_splits[split][partition] = split_idx

        full_dataset = self.datasets["full"]
        for split, partition_splits in overall_splits.items():
            self.datasets[split] = PartitionedSubset(full_dataset, partition_splits)

        self.dataloaders = {}
        heimdall_collate_fn = get_collation_closure(self.tasklist)
        per_device_batch_size = self.local_cfg.trainer.args.batchsize // self.accelerator.num_processes
        self.dataloaders = {
            split: DataLoader(
                dataset,
                batch_sampler=PartitionedBatchSampler(
                    PartitionedDistributedSampler(
                        dataset,
                        num_replicas=self.num_replicas,
                        rank=self.rank,
                        shuffle=self.tasklist.shuffle if split == "train" else False,
                    ),
                    batch_size=per_device_batch_size,
                    drop_last=False,
                    tasklist=self.tasklist,
                ),
                collate_fn=heimdall_collate_fn,
                # num_workers=4,  # TODO: currently doesn't work. To fix, will need to create
                # separate DataLoader for each partition, wrap them all with accelerate,
                # and return accordingly.
            )
            for split, dataset in self.datasets.items()
        }

        dataset_str = pformat(self.datasets).replace("\n", "\n\t")
        self.print_during_setup(
            f"> Finished setting up datasets (and loaders):\n\t{dataset_str}",
            is_printable_process=True,
        )


def setup_accelerator(config, cpu=False, run_wandb=False):
    accelerator_log_kwargs = {}
    if run_wandb:
        accelerator_log_kwargs["log_with"] = "wandb"
        accelerator_log_kwargs["project_dir"] = config.work_dir

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))

    accelerator = Accelerator(
        gradient_accumulation_steps=config.scfm.trainer.args.accumulate_grad_batches,
        step_scheduler_with_optimizer=False,
        cpu=cpu,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        **accelerator_log_kwargs,
    )

    return accelerator


def setup_data(config, cpu=False, accelerator=None):
    """Set up Heimdall data based on config, including cr and accelerator."""
    run_wandb = getattr(config, "run_wandb", False)
    if accelerator is None:
        accelerator = setup_accelerator(config, cpu=cpu, run_wandb=run_wandb)

    if accelerator.is_main_process:
        print(OmegaConf.to_yaml(config, resolve=True))

    with open_dict(config):
        only_preprocess_data = config.pop("only_preprocess_data", None)
        # pop so hash of cfg is not changed depending on value

    cr = instantiate_from_config(config.scfm.trainer.cell_rep_config, config, accelerator)

    return accelerator, cr, run_wandb, only_preprocess_data
