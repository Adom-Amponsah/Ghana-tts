"""Inspect ghanaopendata/ghana-english-tts-filtered without assuming split names.

Known issue: the repo's parquet shards are named `filtered-train-XXXXX-of-00082.parquet`.
The `datasets` library derives the split name `filtered-train` from those filenames,
but split names must match ^\\w+(\\.\\w+)*$ (no hyphens), so the standard
config/split discovery fails. This is also why the HF dataset viewer is broken.

Fallback: stream the parquet files directly via hf:// URLs, bypassing split inference.
"""

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import HfApi

DATASET_ID = "ghanaopendata/ghana-english-tts-filtered"


def main() -> None:
    print(f"Dataset: {DATASET_ID}\n")

    # --- 1. Repo file listing (ground truth) ---
    api = HfApi()
    files = api.list_repo_tree(DATASET_ID, repo_type="dataset", path_in_repo="data", recursive=True)
    paths = sorted(getattr(f, "path", "") for f in files if getattr(f, "path", "").endswith(".parquet"))
    print(f"PARQUET SHARDS IN REPO: {len(paths)}")
    print(f"  first: {paths[0]}")
    print(f"  last:  {paths[-1]}")

    # --- 2. Standard config/split discovery ---
    print("\nCONFIGS (standard discovery)")
    try:
        configs = get_dataset_config_names(DATASET_ID)
        for config in configs:
            print(f"  - {config}")
        for config in configs:
            print(f"\nSPLITS FOR CONFIG: {config}")
            try:
                splits = get_dataset_split_names(DATASET_ID, config_name=config)
                for split in splits:
                    print(f"  - {split}")
            except Exception as exc:
                print(f"  ERROR: {exc}")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        print("\n  -> Falling back to direct parquet streaming (bypasses split inference).")

    # --- 3. Direct parquet streaming: schema + row count + sample row ---
    print("\nDIRECT PARQUET STREAM (first shard)")
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{DATASET_ID}/data/{paths[0].split('/')[-1]}",
        split="train",
        streaming=True,
    )

    first = next(iter(ds))
    print("  Features:")
    for name, feature in first.items():
        value = feature
        if isinstance(value, dict):
            value = {k: (type(v).__name__ if not isinstance(v, (int, float, str)) else v) for k, v in value.items()}
        print(f"    {name}: {value!r}"[:120])


if __name__ == "__main__":
    main()
