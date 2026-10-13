import math

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler, Subset

from Heimdall.datasets import PartitionedDataset, PartitionedSubset


class PartitionedDataLoader:
    """Iterates one epoch across all of a `PartitionedDataset`'s partitions,
    giving each partition its own real `DataLoader` (with `num_workers`
    workers) instead of one `DataLoader` spanning every partition.

    A single dataloader can't have workers here: worker processes are forked
    once (at the start of iteration) and never see the main process's later
    `full_dataset.data.partition = ...` switches -- each fork gets its own
    copy of the dataset object, so they'd keep serving from whichever
    partition happened to be loaded at fork time. Building a fresh,
    short-lived `DataLoader` *after* switching to each partition means its
    workers fork with the right partition's `adata` already loaded, and get
    torn down (along with that partition's data) before the next switch.

    Batches never span a partition boundary here, matching the previous
    single-dataloader behavior (`PartitionedBatchSampler` flushed each
    partition's remainder as its own short final batch rather than
    concatenating it with the next partition's samples).
    """

    def __init__(
        self,
        full_dataset: PartitionedDataset,
        partition_indices: dict,
        batch_size: int,
        shuffle: bool,
        collate_fn,
        num_workers: int,
        num_replicas: int,
        rank: int,
        seed: int,
        drop_last: bool = False,
    ):
        self.full_dataset = full_dataset
        self.data_rep = full_dataset.data
        self.partition_indices = {p: idx for p, idx in partition_indices.items() if len(idx) > 0}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Exposed for callers that only need a total sample count (e.g. LR-scheduler total-step
        # math via `len(dataloader.dataset)`), not for direct indexing.
        self.dataset = PartitionedSubset(full_dataset, partition_indices)

    def __len__(self):
        # Mirrors the previous cross-partition sampler's length: the combined per-rank sample
        # count across *all* partitions, divided into batches once, globally -- not a sum of
        # per-partition batch counts. Those can differ slightly whenever a partition's per-rank
        # sample count isn't a multiple of `batch_size`, since real iteration flushes each
        # partition's remainder as its own shorter final batch. Kept as-is for parity with
        # existing step-counting/logging math, which relies on this exact formula.
        total_per_rank = sum(math.ceil(len(idx) / self.num_replicas) for idx in self.partition_indices.values())
        if self.drop_last:
            return total_per_rank // self.batch_size
        return math.ceil(total_per_rank / self.batch_size)

    def __iter__(self):
        order = list(self.partition_indices)
        if self.shuffle:
            order = np.random.default_rng(seed=self.seed + self.epoch).permutation(order).tolist()

        for partition in order:
            self.data_rep.partition = partition

            # Warm the per-partition CSR row-cache (`utils.py::_read_sparse_row`'s
            # `_heimdall_csr_cache`) here, in this process, before `DataLoader` below forks
            # `num_workers` worker processes. That cache is built lazily on first `__getitem__`,
            # so without this warm-up every forked worker independently rereads the whole
            # partition off disk and builds its own full copy instead of inheriting this one via
            # copy-on-write -- multiplying resident memory by roughly `num_workers * num_replicas`.
            # Confirmed live via `ps`: with this warm-up missing, 4 workers under 1 DDP rank sat
            # at ~26GB RSS each (fit fine); 4 workers x 2 DDP ranks sat at ~47GB RSS each and got
            # OOM-killed (jobs 54643/54644) even at --mem=400Gb, identically regardless of
            # NCCL_P2P_DISABLE -- ruling out NCCL and confirming this per-worker duplication.
            if self.num_workers > 0:
                self.data_rep.fc[0]

            sampler = DistributedSampler(
                Subset(self.full_dataset, self.partition_indices[partition]),
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            sampler.set_epoch(self.epoch)

            loader = DataLoader(
                sampler.dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
            )
            yield from loader

        self.data_rep.partition = None
        self.epoch += 1
