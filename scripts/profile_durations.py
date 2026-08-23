"""Full-corpus duration profile: metadata-only read of duration_ss.

Reads ONLY the duration_ss column chunk from all 88 parquet shards via the
HF fsspec filesystem (no audio, no full shard download). Produces:

  data/manifests/all_durations.csv   shard + duration_ss for every clip
  data/manifests/duration_profile.csv  threshold survival table

and prints hours/clip counts surviving >=8s / >=10s / >=12s floors.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "manifests" / "shard_metadata.csv"
OUT_DIR = ROOT / "data" / "manifests"
DATASET_ID = "ghanaopendata/ghana-english-tts-filtered"
THRESHOLDS = [4, 6, 8, 10, 12, 13, 14]
MAX_WORKERS = 6
RETRIES = 4


def fs_open():
    import fsspec
    return fsspec.filesystem("hf")


def read_shard_durations(fs, shard: str) -> tuple[str, list[float] | None]:
    url = f"hf://datasets/{DATASET_ID}/data/{shard}"
    for attempt in range(1, RETRIES + 1):
        try:
            with fs.open(url, "rb") as f:
                t = pq.read_table(f, columns=["duration_ss"])
            return shard, t.column("duration_ss").to_pylist()
        except Exception as e:  # noqa: BLE001 - retry any transient error
            if attempt == RETRIES:
                print(f"  [FAIL] {shard}: {e}", flush=True)
                return shard, None
            wait = 5 * attempt
            print(f"  [retry {attempt}/{RETRIES - 1}] {shard}: {e} -- wait {wait}s", flush=True)
            time.sleep(wait)
    return shard, None


def main() -> None:
    meta = pd.read_csv(META)
    shards = sorted(meta["shard"])  # keep the .parquet suffix -- URL needs it
    print(f"profiling duration_ss across {len(shards)} shards (metadata-only)...")

    fs = fs_open()
    parts = []
    failed = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(read_shard_durations, fs, s) for s in shards]
        for fut in as_completed(futures):
            shard, durs = fut.result()
            done += 1
            if durs is None:
                failed.append(shard)
            else:
                parts.append(pd.DataFrame({"shard": shard, "duration_ss": durs}))
            if done % 10 == 0 or done == len(shards):
                print(f"  {done}/{len(shards)} shards  ({time.time() - t0:.0f}s)", flush=True)

    if failed:
        print(f"\n[error] {len(failed)} shards failed: {failed}")
        print("        re-run to retry; successful reads are re-fetched (cheap).")
        if not parts:
            sys.exit(1)

    df = pd.concat(parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_durations.csv", index=False)

    n = len(df)
    total_h = df["duration_ss"].sum() / 3600
    print(f"\nclips: {n:,}   total duration: {total_h:.1f} h")
    print(f"min {df['duration_ss'].min():.2f}s  median {df['duration_ss'].median():.2f}s  max {df['duration_ss'].max():.2f}s")

    rows = []
    for thr in [0] + THRESHOLDS:
        sub = df[df["duration_ss"] >= thr]
        rows.append({
            "floor_s": thr,
            "clips": len(sub),
            "clips_pct": round(100 * len(sub) / n, 1),
            "hours": round(sub["duration_ss"].sum() / 3600, 1),
            "hours_pct": round(100 * sub["duration_ss"].sum() / df["duration_ss"].sum(), 1),
        })
    prof = pd.DataFrame(rows)
    prof.to_csv(OUT_DIR / "duration_profile.csv", index=False)

    print("\nsurvival by duration floor:")
    print(prof.to_string(index=False))
    print(f"\n[done] -> {OUT_DIR / 'all_durations.csv'}")
    print(f"[done] -> {OUT_DIR / 'duration_profile.csv'}")


if __name__ == "__main__":
    sys.exit(main())
