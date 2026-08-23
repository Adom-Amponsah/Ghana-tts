"""Resume/finish the 200-clip sample after the interrupted run.

State on disk: 153 clips (idx 000-152) from shards 00000, 00001, 00002,
00003, 00024, 00045. The final manifest only needs 40 clips per stratum, so
the remaining slots are filled from untouched shards (00066, 00087) and no
completed shard is ever re-downloaded.

Strategy:
  - Scan existing WAVs -> per-stratum counts.
  - Completed shards: rebuild manifest rows with a metadata-only remote read
    (parquet column projection -> only the tiny text/float columns transfer,
    never the audio).
  - Untouched shards: download with the `hf` CLI (resumable, robust retries),
    extract only the clips needed to fill per-stratum deficits, then delete
    the local shard file.
"""

import io
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

DATASET_ID = "ghanaopendata/ghana-english-tts-filtered"
ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SAMPLES = ROOT / "data" / "samples"
SHARD_DIR = ROOT / "data" / "shards"
SHARD_META_CSV = MANIFESTS / "shard_metadata.csv"
SAMPLE_CSV = MANIFESTS / "sample_200.csv"
HF_EXE = ROOT / ".venv" / "Scripts" / "hf.exe"

SEED = 42
N_PER_STRATUM = 40
N_SHARDS = 8
N_BORDERLINE_SHARDS = 3
META_COLS = ["duration_ss", "mean_speech_prob", "dbfs"]
TEXT_COLS = META_COLS + ["corrected_text"]

STRATA = ["short", "medium", "long", "hi_quality", "borderline"]
PRIORITY = ["short", "medium", "long", "hi_quality", "borderline"]
TARGET = {st: N_PER_STRATUM for st in STRATA}

WAV_RE = re.compile(
    r"^(\d{3})_(short|medium|long|hi_quality|borderline)_"
    r"(filtered-train-\d{5}-of-\d{5})_r(\d+)\.wav$"
)

# functions shared with build_sample.py ------------------------------------


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
        prob_cut = max(meta["mean_speech_prob_min"].quantile(0.30), df["mean_speech_prob"].quantile(0.10))
        dbfs_cut = max(meta["dbfs_min"].quantile(0.30), df["dbfs"].quantile(0.10))
        return (df["mean_speech_prob"] <= prob_cut) | (df["dbfs"] <= dbfs_cut)
    raise ValueError(stratum)


def pick_shards(meta: pd.DataFrame) -> list[str]:
    prob_cut = meta["mean_speech_prob_min"].quantile(0.30)
    dbfs_cut = meta["dbfs_min"].quantile(0.30)
    tail = meta[(meta["mean_speech_prob_min"] <= prob_cut) | (meta["dbfs_min"] <= dbfs_cut)]
    tail = tail.sort_values("mean_speech_prob_min")
    border = tail["shard"].head(N_BORDERLINE_SHARDS).tolist()
    rest = meta[~meta["shard"].isin(border)]["shard"].tolist()
    spread = [rest[int(i * (len(rest) - 1) / 4)] for i in range(5)]
    return border + spread


def assign_slots(meta: pd.DataFrame, shards: list[str]) -> dict[str, list[str]]:
    from collections import deque

    assignments: dict[str, list[str]] = {s: [] for s in shards}
    shard_rows = meta.set_index("shard")
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
            if pending[st] > 0 and assigned_here[pair] < 30:
                queue.append(pair)
            if sum(pending.values()) == 0:
                break
        if not progressed:
            raise RuntimeError("assignment stuck")
    return assignments


# resume logic ---------------------------------------------------------------


def scan_existing() -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for p in sorted(SAMPLES.glob("*.wav")):
        m = WAV_RE.match(p.name)
        if not m:
            print(f"  [warn] unparseable wav: {p.name}")
            continue
        found.setdefault(m.group(3) + ".parquet", []).append({
            "sample_index": int(m.group(1)),
            "stratum": m.group(2),
            "shard": m.group(3),
            "row_index": int(m.group(4)),
            "wav_file": p.name,
        })
    return found


def hf_download_shard(shard: str) -> Path:
    """Resumable download of one shard via the hf CLI."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(HF_EXE), "download", DATASET_ID,
        "--repo-type", "dataset",
        "--include", f"data/{shard}",
        "--local-dir", str(SHARD_DIR),
    ]
    print(f"[download] {shard}")
    subprocess.run(cmd, check=True)
    return SHARD_DIR / "data" / shard


def main() -> None:
    random.seed(SEED)
    meta = pd.read_csv(SHARD_META_CSV).sort_values("shard").reset_index(drop=True)
    offsets = dict(zip(meta["shard"], meta["num_rows"].cumsum() - meta["num_rows"]))

    picked = pick_shards(meta)
    existing = scan_existing()
    total_existing = sum(len(v) for v in existing.values())
    print(f"[scan] {total_existing} clips already on disk")

    have = Counter(e["stratum"] for v in existing.values() for e in v)
    deficit = {st: TARGET[st] - have.get(st, 0) for st in STRATA}
    print(f"[have]    {dict(have)}")
    print(f"[deficit] {deficit}  (total {sum(deficit.values())})")

    # Shards with no clips yet are the only ones we download.
    download_shards = [s for s in picked if s not in existing]
    print(f"[download] untouched shards: {download_shards}")

    next_idx = max((e["sample_index"] for v in existing.values() for e in v), default=-1) + 1
    records = []
    fs = fsspec.filesystem("hf")

    # ---------- completed shards: metadata-only remote read ----------
    for shard in picked:
        if shard not in existing:
            continue
        rows = existing[shard]
        print(f"[meta] {shard}: fetching metadata for {len(rows)} existing clips (no audio)...")
        with fs.open(f"hf://datasets/{DATASET_ID}/data/{shard}", "rb") as f:
            df = pq.read_table(f, columns=TEXT_COLS).to_pandas()
        for e in rows:
            row = df.loc[e["row_index"]]
            sr = sf.info(SAMPLES / e["wav_file"]).samplerate
            records.append({
                **e,
                "global_row": offsets[shard] + e["row_index"],
                "wav_sample_rate": sr,
                "duration_ss": row["duration_ss"],
                "mean_speech_prob": row["mean_speech_prob"],
                "dbfs": row["dbfs"],
                "corrected_text": row["corrected_text"],
            })

    # ---------- untouched shards: fill the deficits ----------
    for shard in tqdm(download_shards, desc="shards"):
        if sum(deficit.values()) == 0:
            break
        local = hf_download_shard(shard)
        table = pq.read_table(local, columns=TEXT_COLS + ["audio"])
        df = table.to_pandas()
        used_rows: set[int] = set()

        # duration bands first (disjoint masks), then quality strata
        for stratum in PRIORITY:
            while deficit[stratum] > 0:
                mask = row_mask(df, stratum, meta) & ~df.index.isin(used_rows)
                candidates = df.index[mask].tolist()
                if not candidates:
                    tqdm.write(f"  [!] {shard}: no more rows for {stratum}, moving on")
                    break
                ridx = int(random.Random(SEED + next_idx).choice(candidates))
                used_rows.add(ridx)
                row = df.loc[ridx]
                wav_name = f"{next_idx:03d}_{stratum}_{shard.replace('.parquet', '')}_r{ridx}.wav"
                data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                sf.write(SAMPLES / wav_name, data, sr, subtype="PCM_16")
                records.append({
                    "sample_index": next_idx,
                    "stratum": stratum,
                    "shard": shard,
                    "row_index": ridx,
                    "global_row": offsets[shard] + ridx,
                    "wav_file": wav_name,
                    "wav_sample_rate": sr,
                    "duration_ss": row["duration_ss"],
                    "mean_speech_prob": row["mean_speech_prob"],
                    "dbfs": row["dbfs"],
                    "corrected_text": row["corrected_text"],
                })
                next_idx += 1
                deficit[stratum] -= 1
        del table, df
        local.unlink()  # free the 1.57 GB once extracted
        tqdm.write(f"[done] {shard} extracted, local file removed; deficit now {deficit}")

    if sum(deficit.values()) > 0:
        print(f"[warn] unfilled deficits remain: {deficit}")

    out = pd.DataFrame(records).sort_values("sample_index")
    out.to_csv(SAMPLE_CSV, index=False)
    print(f"\n[finished] {len(out)} clips -> {SAMPLE_CSV}")
    per = out.groupby("stratum").size()
    print("per stratum:", dict(per))
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
