"""Sample a leakage-free ~20M-cell pretraining corpus from the full merged pool.

Draws a random subset of `--target-n` cells (default 20,000,000) from the
~22.19M-cell merged pool at `merged_data.zarr`, explicitly excluding every
cell held out (val/test) for the scTab GI-tract `split1` benchmark -- so this
new pretraining corpus is guaranteed never to have seen any of that
benchmark's evaluation cells, unlike the existing 10k/100k/1M subsets (which
were sampled without any such exclusion).

`X` in the source Zarr store is dense, so this reads it directly (no Dask
needed) in row-blocks, subsets each block down to just the selected rows for
this sample, converts to sparse CSR (compact on disk -- the sparse-backed
per-cell read path in `Heimdall.utils._get_inputs_from_csr` now reads rows
directly from the raw indptr/indices/data arrays, so this works correctly at
any scale), and writes out fixed-size shards -- matching the
directory-of-shards layout `PartitionedCellRepresentation` already expects.

Usage:
    python sample_20mill_clean.py \
        --out-dir /work/magroup/shared/Heimdall/pretrain_data/pretrain_subsets/subset_20mill_clean
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from scipy.sparse import csr_matrix, vstack

MERGED_BASE_DIR = Path("/work/magroup/nzh/scTab/notebooks/store_creation")
GI_PATH = Path("/work/magroup/shared/Heimdall/data/sctab/tissue_splits_spencer/scTab_GItract_train.h5ad")
SPLIT_COL = "split1"
HOLDOUT_LABELS = ("val", "test")


def get_excluded_ids() -> set:
    """soma_joinids held out (val/test) for the scTab GI-tract split1 benchmark."""
    gi = ad.read_h5ad(GI_PATH, backed="r")
    holdout_mask = gi.obs[SPLIT_COL].isin(HOLDOUT_LABELS)
    return set(gi.obs.loc[holdout_mask, "soma_joinid"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to write shard .h5ad files into")
    parser.add_argument("--target-n", type=int, default=20_000_000, help="number of cells to sample")
    parser.add_argument("--shard-size", type=int, default=1_000_000, help="approx. cells per output shard")
    parser.add_argument("--block-size", type=int, default=20_000, help="rows read from zarr at a time")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="threads concurrently reading row-blocks from the zarr store (I/O-bound, GIL released during "
        "blosc decompress, so this overlaps per-chunk-file read latency instead of serializing it)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading merged obs/var metadata...")
    merged_obs = pd.read_parquet(MERGED_BASE_DIR / "merged_obs.parquet")
    merged_var = pd.read_parquet(MERGED_BASE_DIR / "merged_var.parquet")
    n_obs = len(merged_obs)
    print(f"Merged pool: {n_obs:,} x {len(merged_var):,}")

    print("Collecting excluded (held-out benchmark) soma_joinids...")
    excluded_ids = get_excluded_ids()
    print(f"  {len(excluded_ids):,} cells to exclude")

    eligible_mask = ~merged_obs["soma_joinid"].isin(excluded_ids).to_numpy()
    eligible_positions = np.nonzero(eligible_mask)[0]
    print(f"  {len(eligible_positions):,} eligible cells remain after exclusion")

    if args.target_n > len(eligible_positions):
        raise ValueError(
            f"--target-n={args.target_n:,} exceeds the {len(eligible_positions):,} eligible cells available.",
        )

    rng = np.random.default_rng(args.seed)
    selected_positions = rng.choice(eligible_positions, size=args.target_n, replace=False)
    selected_positions.sort()  # ascending, for efficient sequential zarr reads
    print(f"Sampled {len(selected_positions):,} cells for the new pretraining corpus")

    print(f"Opening {MERGED_BASE_DIR / 'merged_data.zarr'} (read-only)...")
    root_store = zarr.DirectoryStore(str(MERGED_BASE_DIR / "merged_data.zarr"))
    root_group = zarr.open_group(root_store, mode="r")
    X_zarr = root_group["X"]

    shard_chunks = []  # list of (csr_matrix, positions) accumulated for the current shard
    shard_cell_count = 0
    shard_idx = 0
    num_digits = len(str(-(-args.target_n // args.shard_size)))

    def flush_shard():
        nonlocal shard_chunks, shard_cell_count, shard_idx
        if not shard_chunks:
            return

        shard_X = vstack([chunk_X for chunk_X, _ in shard_chunks], format="csr")
        shard_positions = np.concatenate([chunk_pos for _, chunk_pos in shard_chunks])
        shard_obs = merged_obs.iloc[shard_positions].copy()

        out_path = args.out_dir / f"shard_{shard_idx:0{num_digits}d}.h5ad"
        shard_adata = ad.AnnData(X=shard_X, obs=shard_obs, var=merged_var.copy())
        shard_adata.write_h5ad(out_path)
        print(f"  wrote {out_path} ({shard_X.shape[0]:,} cells)")

        shard_chunks = []
        shard_cell_count = 0
        shard_idx += 1

    print("Planning row-blocks...")
    n_selected = len(selected_positions)
    block_plan = []  # (block_start, block_end, block_selected)
    ptr = 0
    block_start = 0
    while block_start < n_obs and ptr < n_selected:
        block_end = min(block_start + args.block_size, n_obs)
        lo = np.searchsorted(selected_positions, block_start, side="left")
        hi = np.searchsorted(selected_positions, block_end, side="left")
        block_selected = selected_positions[lo:hi]
        if len(block_selected) > 0:
            block_plan.append((block_start, block_end, block_selected))
            ptr = hi
        block_start = block_end
    print(f"  {len(block_plan):,} non-empty blocks to read (of {-(-n_obs // args.block_size):,} total)")

    def read_block(block):
        block_start, block_end, block_selected = block
        block_dense = X_zarr[block_start:block_end, :]
        local_rows = block_selected - block_start
        chunk_dense = block_dense[local_rows, :]
        return csr_matrix(chunk_dense), block_selected

    print(f"Streaming selected rows out of the Zarr store with {args.num_workers} reader threads...")
    ptr = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for num_blocks_done, (chunk_X, block_selected) in enumerate(
            executor.map(read_block, block_plan, chunksize=4),
            start=1,
        ):
            shard_chunks.append((chunk_X, block_selected))
            shard_cell_count += len(block_selected)
            ptr += len(block_selected)

            if shard_cell_count >= args.shard_size:
                flush_shard()

            if num_blocks_done % 50 == 0:
                print(f"  progress: read {num_blocks_done:,}/{len(block_plan):,} blocks, selected {ptr:,}/{n_selected:,}")

    flush_shard()  # flush any remainder
    print("Done.")
    print(f"\nNext step: point dataset.preprocess_args.data_path at {args.out_dir}")


if __name__ == "__main__":
    main()
