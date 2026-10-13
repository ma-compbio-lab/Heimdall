"""Split a single large `.h5ad` file into row-contiguous shards.

Why: Heimdall's per-cell dataloader indexes into a *backed* sparse `adata.X` one
cell at a time via a length-1 fancy index (`adata.X[[cell_index], :]`), which
breaks for datasets where `n_obs * n_vars` exceeds the int32 range (~2.147e9) --
see the `AttributeError: 'tuple' object has no attribute 'ndim'` crashes on the
1M-cell pretraining runs.

A first version of this script tried to work around that by reading contiguous
*slices* of the backed AnnData (`adata[start:end]`), on the assumption that slice
objects take a different, safe code path in scipy's sparse indexing. That still
failed (`TypeError: '>=' not supported between instances of 'int' and 'NoneType'`)
because anndata's own backed-subsetting machinery (`subset_sparsedataset` ->
`BaseCompressedSparseDataset.__getitem__`) converts the slice into a fancy index
internally before scipy ever sees it, so it still routes through the same fragile
large-sparse-array code as before.

This version avoids AnnData's/scipy's indexing machinery entirely. A CSR matrix
stores every row's nonzeros contiguously, so extracting rows [start:end) is just:
  - `indptr[start:end+1]` for the new row pointers (rebased to start at 0)
  - `data[indptr[start]:indptr[end]]` and `indices[indptr[start]:indptr[end]]`
    for the corresponding nonzero values/columns
These are plain contiguous h5py dataset reads (hyperslab reads) -- no fancy
indexing, no int32-vs-shape overflow risk, guaranteed to work regardless of the
source matrix's total size. We read those raw arrays directly via h5py, build a
fresh `scipy.sparse.csr_matrix` per shard, and wrap it in a new AnnData with the
corresponding `obs` rows and the same `var`/`uns`.

Usage:
    python split_1mill_shards.py \
        --source /work/magroup/shared/Heimdall/pretrain_data/pretrain_subsets/subset_1mill/merged_1M.h5ad \
        --out-dir /work/magroup/shared/Heimdall/pretrain_data/pretrain_subsets/subset_1mill_sharded \
        --num-shards 10
"""
import argparse
from pathlib import Path

import anndata as ad
import h5py
from scipy.sparse import csr_matrix

INT32_MAX = 2**31 - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Path to the single merged .h5ad file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to write shard .h5ad files into")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of shards to split into")
    parser.add_argument("--prefix", default="shard", help="Filename prefix for shard files")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading obs/var/uns metadata from {args.source} (backed)...")
    meta = ad.read_h5ad(args.source, backed="r")
    n_obs, n_vars = meta.shape
    obs = meta.obs
    var = meta.var
    uns = dict(meta.uns)
    print(f"Source shape: {n_obs:,} x {n_vars:,}  (n_obs*n_vars = {n_obs * n_vars:,})")

    max_rows_per_shard = INT32_MAX // n_vars
    print(f"Max safe rows/shard for {n_vars:,} genes: {max_rows_per_shard:,}")

    shard_size = -(-n_obs // args.num_shards)  # ceil division
    if shard_size > max_rows_per_shard:
        min_shards = -(-n_obs // max_rows_per_shard)  # ceil division
        raise ValueError(
            f"--num-shards={args.num_shards} gives {shard_size:,} rows/shard, which exceeds "
            f"the safe limit of {max_rows_per_shard:,} rows for {n_vars:,} genes. "
            f"Use at least --num-shards={min_shards}.",
        )
    print(f"Using shard_size={shard_size:,} rows across {args.num_shards} shards")

    num_digits = len(str(args.num_shards - 1))

    with h5py.File(args.source, "r") as f:
        X = f["X"]
        encoding = X.attrs.get("encoding-type")
        if encoding != "csr_matrix":
            raise ValueError(
                f"Expected X to be encoded as 'csr_matrix' for this row-slicing approach, got {encoding!r}.",
            )
        indptr_full = X["indptr"][:]  # shape (n_obs+1,), small -- fine to load fully

        for i in range(args.num_shards):
            start = i * shard_size
            end = min(start + shard_size, n_obs)
            if start >= end:
                break

            out_path = args.out_dir / f"{args.prefix}_{i:0{num_digits}d}.h5ad"
            if out_path.exists():
                print(f"  [{i}] {out_path} already exists, skipping")
                continue

            data_start, data_end = int(indptr_full[start]), int(indptr_full[end])
            print(f"  [{i}] rows [{start:,}:{end:,})  nnz_range=[{data_start:,}:{data_end:,}) -> {out_path}")

            data_slice = X["data"][data_start:data_end]
            indices_slice = X["indices"][data_start:data_end]
            new_indptr = indptr_full[start : end + 1] - indptr_full[start]

            shard_X = csr_matrix(
                (data_slice, indices_slice, new_indptr),
                shape=(end - start, n_vars),
            )

            shard_adata = ad.AnnData(
                X=shard_X,
                obs=obs.iloc[start:end].copy(),
                var=var.copy(),
                uns=dict(uns),
            )
            shard_adata.write_h5ad(out_path)

    print("Done.")
    print(f"\nNext step: point dataset.preprocess_args.data_path at {args.out_dir}")


if __name__ == "__main__":
    main()
