"""Build a stratified 200-clip inspection sample (Step 5). Budget: 8 shards.

Each parquet shard is a single ~1.57 GB row group, so any clip extraction costs
a full shard download. Strategy:

  Phase 1 (harvest): read every shard's parquet FOOTER only (a few KB each).
          Footers carry per-column min/max statistics -> per-shard
          duration_ss / mean_speech_prob / dbfs ranges + exact row counts.
          No audio bytes are transferred.
  Phase 2 (select):  pick 8 shards:
            - the 3 shards owning the lower tail of mean_speech_prob / dbfs
              (so the "borderline" stratum sees what the filter allowed through)
            - 5 more spread evenly across the 0-87 index range
          Assign 200 clip slots (5 strata x 40) to shards round-robin,
          honoring per-stratum eligibility from the footer stats.
  Phase 3 (fetch):  download each selected shard ONCE (~1.57 GB each),
          extract its assigned clips, save WAVs (16 kHz mono PCM16).
  Phase 4 (report): manifest + statistics.

Outputs:
  data/manifests/shard_metadata.csv   (footer harvest, all 88 shards)
  data/manifests/sample_200.csv       (one row per sampled clip)
  data/samples/*.wav
"""

import io
import random
import sys
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi
from tqdm import tqdm

DATASET_ID = "ghanaopendata/ghana-english-tts-filtered"
ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SAMPLES = ROOT / "data" / "samples"
SHARD_META_CSV = MANIFESTS / "shard_metadata.csv"
SAMPLE_CSV = MANIFESTS / "sample_200.csv"

SEED = 42
N_PER_STRATUM = 40
N_SHARDS = 8
N_BORDERLINE_SHARDS = 3
META_COLS = ["duration_ss", "mean_speech_prob", "dbfs"]

STRATA = ["short", "medium", "long", "hi_quality", "borderline"]
STRATA_DESC = {
    "short": "duration 2-6 s",
    "medium": "duration 6-10 s",
    "long": "duration 10-15 s",
    "hi_quality": "mean_speech_prob >= 0.95",
    "borderline": "low-end mean_speech_prob / dbfs",
}


def list_shards() -> list[str]:
    api = HfApi()
    files = api.list_repo_tree(DATASET_ID, repo_type="dataset", path_in_repo="data", recursive=True)
    return sorted(f"data/{f.path.split('/')[-1]}" for f in files if f.path.endswith(".parquet"))


def harvest_footer(fs, shard: str) -> dict:
    """Read only the parquet footer: row count + per-column min/max stats."""
    with fs.open(f"hf://datasets/{DATASET_ID}/{shard}", "rb") as f:
        md = pq.ParquetFile(f).metadata
    assert md.num_row_groups == 1, f"{shard}: expected 1 row group"
    rg = md.row_group(0)
    row = {"shard": shard.split("/")[-1], "num_rows": rg.num_rows}
    for c in range(rg.num_columns):
        col = rg.column(c)
        if col.path_in_schema in META_COLS and col.is_stats_set:
            row[f"{col.path_in_schema}_min"] = col.statistics.min
            row[f"{col.path_in_schema}_max"] = col.statistics.max
    return row


def eligible(stratum: str, shard_row: pd.Series, meta: pd.DataFrame) -> bool:
    if stratum == "short":
        return bool(shard_row["duration_ss_min"] <= 6.0)
    if stratum == "medium":
        return bool(shard_row["duration_ss_min"] <= 10.0 and shard_row["duration_ss_max"] >= 6.0)
    if stratum == "long":
        return bool(shard_row["duration_ss_max"] >= 10.0)
    if stratum == "hi_quality":
        return bool(shard_row["mean_speech_prob_max"] >= 0.95)
    if stratum == "borderline":
        prob_cut = meta["mean_speech_prob_min"].quantile(0.30)
        dbfs_cut = meta["dbfs_min"].quantile(0.30)
        return bool(shard_row["mean_speech_prob_min"] <= prob_cut or shard_row["dbfs_min"] <= dbfs_cut)
    raise ValueError(stratum)


def row_mask(df: pd.DataFrame, stratum: str, meta: pd.DataFrame) -> pd.Series:
    if stratum == "short":
        return df["duration_ss"].between(2.0, 6.0)
    if stratum == "medium":
        return df["duration_ss"].between(6.0, 10.0)
    if stratum == "long":
        return df["duration_ss"].between(10.0, 15.0)
    if stratum == "hi_quality":
        return df["mean_speech_prob"] >= 0.95
    if stratum == "borderline":
        # Lower end of the AVAILABLE distribution: global footer-tail cut,
        # relaxed to this shard's own bottom decile if the shard is cleaner.
        prob_cut = max(meta["mean_speech_prob_min"].quantile(0.30), df["mean_speech_prob"].quantile(0.10))
        dbfs_cut = max(meta["dbfs_min"].quantile(0.30), df["dbfs"].quantile(0.10))
        return (df["mean_speech_prob"] <= prob_cut) | (df["dbfs"] <= dbfs_cut)
    raise ValueError(stratum)


def pick_shards(meta: pd.DataFrame) -> list[str]:
    """3 lower-tail shards + 5 spread evenly across the index range."""
    prob_cut = meta["mean_speech_prob_min"].quantile(0.30)
    dbfs_cut = meta["dbfs_min"].quantile(0.30)
    tail = meta[(meta["mean_speech_prob_min"] <= prob_cut) | (meta["dbfs_min"] <= dbfs_cut)]
    tail = tail.sort_values("mean_speech_prob_min")
    border = tail["shard"].head(N_BORDERLINE_SHARDS).tolist()

    rest = meta[~meta["shard"].isin(border)]["shard"].tolist()
    spread = [rest[int(i * (len(rest) - 1) / 4)] for i in range(5)]
    return border + spread


def assign_slots(meta: pd.DataFrame, shards: list[str]) -> dict[str, list[str]]:
    """Round-robin 200 slots (5 strata x 40) over shards, honoring eligibility."""
    from collections import deque

    assignments: dict[str, list[str]] = {s: [] for s in shards}
    shard_rows = meta.set_index("shard")

    # Every (shard, stratum) pair that can actually provide clips, shuffled
    # so the round-robin doesn't favor the first shard/stratum.
    pairs = [(s, st) for s in shards for st in STRATA if eligible(st, shard_rows.loc[s], meta)]
    random.Random(SEED).shuffle(pairs)
    queue = deque(pairs)

    pending = {st: N_PER_STRATUM for st in STRATA}
    assigned_here = {p: 0 for p in pairs}

    while sum(pending.values()) > 0:
        progressed = False
        for _ in range(len(queue)):
            pair = queue.popleft()
            shard, st = pair
            if pending[st] > 0:
                assignments[shard].append(st)
                pending[st] -= 1
                assigned_here[pair] += 1
                progressed = True
            # Keep cycling pairs until their stratum is filled, but cap the
            # per-shard load so one shard doesn't absorb everything.
            if pending[st] > 0 and assigned_here[pair] < 30:
                queue.append(pair)
            if sum(pending.values()) == 0:
                break
        if not progressed:
            stuck = [st for st, n in pending.items() if n > 0]
            raise RuntimeError(f"no eligible shard for strata: {stuck}")
    return assignments


def main() -> None:
    random.seed(SEED)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("hf")

    # ---------- Phase 1: harvest footers ----------
    if SHARD_META_CSV.exists():
        print(f"[harvest] reusing {SHARD_META_CSV.name}")
        meta = pd.read_csv(SHARD_META_CSV)
    else:
        shards = list_shards()
        print(f"[harvest] reading footers of {len(shards)} shards...")
        rows = [harvest_footer(fs, s) for s in tqdm(shards)]
        meta = pd.DataFrame(rows)
        meta.to_csv(SHARD_META_CSV, index=False)
    print(f"[harvest] total rows: {meta['num_rows'].sum():,} across {len(meta)} shards")

    # Global row offsets assuming shards are contiguous in filename order.
    meta = meta.sort_values("shard").reset_index(drop=True)
    offsets = dict(zip(meta["shard"], meta["num_rows"].cumsum() - meta["num_rows"]))

    # ---------- Phase 2: select shards + assign slots ----------
    picked = pick_shards(meta)
    print(f"[select] shards ({len(picked)}):")
    for s in picked:
        r = meta.set_index("shard").loc[s]
        print(f"  {s}  rows={r['num_rows']}  "
              f"prob_min={r['mean_speech_prob_min']:.3f}  dbfs_min={r['dbfs_min']:.1f}")
    assignments = assign_slots(meta, picked)
    for s, slots in assignments.items():
        print(f"  {s}: {len(slots)} clips  {sorted(set(slots))}")

    # ---------- Phase 3: fetch shards, extract clips ----------
    records = []
    idx = 0
    for shard in tqdm(picked, desc="shards"):
        slots = assignments[shard]
        if not slots:
            continue
        tqdm.write(f"[fetch] {shard} ({len(slots)} clips)...")
        with fs.open(f"hf://datasets/{DATASET_ID}/data/{shard}", "rb") as f:
            table = pq.read_table(f, columns=META_COLS + ["corrected_text", "audio"])
        df = table.to_pandas()
        shard_name = shard.split("/")[-1]
        used_rows: set[int] = set()

        for stratum in slots:
            mask = row_mask(df, stratum, meta) & ~df.index.isin(used_rows)
            candidates = df.index[mask].tolist()
            if not candidates:
                tqdm.write(f"  [!] {shard_name}: no rows for {stratum}, skipping")
                continue
            ridx = int(random.Random(SEED + idx).choice(candidates))
            used_rows.add(ridx)
            row = df.loc[ridx]

            wav_name = f"{idx:03d}_{stratum}_{shard_name.replace('.parquet', '')}_r{ridx}.wav"
            data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            sf.write(SAMPLES / wav_name, data, sr, subtype="PCM_16")

            records.append({
                "sample_index": idx,
                "stratum": stratum,
                "shard": shard_name,
                "row_index": ridx,
                "global_row": offsets.get(shard_name, -1) + ridx,
                "wav_file": wav_name,
                "wav_sample_rate": sr,
                "duration_ss": row["duration_ss"],
                "mean_speech_prob": row["mean_speech_prob"],
                "dbfs": row["dbfs"],
                "corrected_text": row["corrected_text"],
            })
            idx += 1
        del table, df

    # ---------- Phase 4: manifest + statistics ----------
    out = pd.DataFrame(records).sort_values("sample_index")
    out.to_csv(SAMPLE_CSV, index=False)
    print(f"\n[done] {len(out)} clips -> {SAMPLE_CSV}")
    print_summary(meta, out)


def print_summary(meta: pd.DataFrame, sample: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("CORPUS METADATA (parquet footer statistics, 88 shards)")
    print("=" * 70)
    print(f"total rows: {meta['num_rows'].sum():,}")
    for col in META_COLS:
        lo, hi = meta[f"{col}_min"], meta[f"{col}_max"]
        print(f"{col:18s} shard-min range [{lo.min():.4f}, {lo.max():.4f}]  "
              f"shard-max range [{hi.min():.4f}, {hi.max():.4f}]")

    print("\n" + "=" * 70)
    print("SAMPLE MANIFEST STATISTICS")
    print("=" * 70)
    print(f"clips: {len(sample)}   shards covered: {sample['shard'].nunique()}/88")
    print("\nper stratum:")
    agg = sample.groupby("stratum").agg(
        n=("sample_index", "size"),
        dur_mean=("duration_ss", "mean"),
        dur_range=("duration_ss", lambda s: f"{s.min():.1f}-{s.max():.1f}"),
        prob_mean=("mean_speech_prob", "mean"),
        dbfs_mean=("dbfs", "mean"),
    )
    print(agg.to_string())

    print("\nfull sample distributions:")
    print(sample[["duration_ss", "mean_speech_prob", "dbfs"]].describe().to_string())

    print("\nduration histogram:")
    bins = [0, 2, 4, 6, 8, 10, 12, 15, 99]
    print(pd.cut(sample["duration_ss"], bins=bins).value_counts().sort_index().to_string())


if __name__ == "__main__":
    sys.exit(main())
