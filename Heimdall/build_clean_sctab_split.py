"""Build a leakage-free `split1_clean` column for the scTab GI-tract benchmark.

Context: `scTab_GItract_train.h5ad` is a subset of the same ~22.19M-cell merged
pool that the 10k/100k/1M pretraining subsets were randomly sampled from,
without excluding this benchmark's held-out (val/test) cells. That means some
of the benchmark's val/test cells were already seen (unlabeled) during
pretraining at each of those scales.

For a fair cross-scale finetuning comparison (10k vs 100k vs 1M vs a future,
deliberately-clean ~20M run), every checkpoint needs to be evaluated on the
*same* held-out set, one that excludes every cell that leaked into *any* of
the pretraining scales being compared. This script:

  1. Collects the union of `soma_joinid`s across the 10k/100k/1M pretraining
     subsets (auto-discovering shard files under each subset directory).
  2. Marks any `split1` val/test cell that's in that union as "excluded" in a
     new `split1_clean` obs column (train-split cells are left untouched --
     overlap with train is expected/fine, only held-out contamination matters).
  3. Writes out `scTab_GItract_train_clean.h5ad` with the new column added.

Usage:
    python build_clean_sctab_split.py
"""
from pathlib import Path

import anndata as ad

GI_PATH = Path("/work/magroup/shared/Heimdall/data/sctab/tissue_splits_spencer/scTab_GItract_train.h5ad")
GI_CLEAN_PATH = GI_PATH.with_name("scTab_GItract_train_clean.h5ad")

PRETRAIN_DIRS = {
    "10k": Path("/work/magroup/shared/Heimdall/pretrain_data/pretrain_subsets/subset_10k"),
    "100k": Path("/work/magroup/shared/Heimdall/pretrain_data/pretrain_subsets/subset_100k"),
    "1M": Path("/work/magroup/shared/Heimdall/pretrain_data/pretrain_subsets/subset_1mill"),
}

SPLIT_COL = "split1"
HOLDOUT_LABELS = ("val", "test")


def collect_pretrained_ids() -> set:
    """Union of soma_joinids across every shard of every pretraining subset."""
    seen_ids = set()
    for name, directory in PRETRAIN_DIRS.items():
        shard_paths = sorted(directory.glob("*.h5ad"))
        if not shard_paths:
            raise FileNotFoundError(f"No .h5ad files found under {directory}")

        subset_ids = set()
        for shard_path in shard_paths:
            shard = ad.read_h5ad(shard_path, backed="r")
            subset_ids.update(shard.obs["soma_joinid"])

        print(f"  {name}: {len(shard_paths)} shard(s), {len(subset_ids):,} cells")
        seen_ids |= subset_ids

    print(f"Union of cells seen across 10k+100k+1M: {len(seen_ids):,}")
    return seen_ids


def main():
    print("Collecting pretrained-cell IDs across 10k/100k/1M subsets...")
    seen_ids = collect_pretrained_ids()

    print(f"\nLoading {GI_PATH} ...")
    gi = ad.read_h5ad(GI_PATH)  # full load, since we're writing a modified copy back out

    clean_col = f"{SPLIT_COL}_clean"
    gi.obs[clean_col] = gi.obs[SPLIT_COL].astype(str)

    is_leaked_holdout = gi.obs["soma_joinid"].isin(seen_ids) & gi.obs[SPLIT_COL].isin(HOLDOUT_LABELS)
    gi.obs.loc[is_leaked_holdout, clean_col] = "excluded"

    print(f"\nExcluded {is_leaked_holdout.sum():,} leaked held-out cells from {clean_col}")
    print(gi.obs[clean_col].value_counts())

    gi.write_h5ad(GI_CLEAN_PATH)
    print(f"\nWrote cleaned file to {GI_CLEAN_PATH}")


if __name__ == "__main__":
    main()
