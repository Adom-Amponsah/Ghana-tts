"""Pilot dataset: ~15 h of >=12 s clips for the architecture bake-off.

Same proven flow as pass 2 (random shards, resumable hf CLI download,
extract, delete shard), scaled up: 8 fresh shards (excluding all 15 audit
shards), ~495 uniform-random >=12 s clips each, ~3,900 clips / ~15 h total.

Outputs:
  data/pilot/wav/*.wav                pl_{idx:05d}_{shardstem}_r{row}.wav
  data/manifests/pilot.csv            wav_file, text, duration, shard
  data/manifests/pilot_f5.txt         F5-TTS fine-tune format: audio|text|duration

Re-runnable: existing WAVs are exempted; only missing clips are downloaded.
"""

import io
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

DATASET_ID = "ghanaopendata/ghana-english-tts-filtered"
ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
PILOT = ROOT / "data" / "pilot"
WAVS = PILOT / "wav"
SHARD_DIR = PILOT / "shards_tmp"
SHARD_META_CSV = MANIFESTS / "shard_metadata.csv"
PILOT_CSV = MANIFESTS / "pilot.csv"
PILOT_F5 = MANIFESTS / "pilot_f5.txt"
HF_EXE = ROOT / ".venv" / "Scripts" / "hf.exe"

SEED = 2026                   # new seed family (audit: 42, 777)
N_SHARDS = 8
TARGET_HOURS = 15.0
MIN_DUR = 12.0                # the confirmed training filter
META_COLS = ["duration_ss", "mean_speech_prob", "dbfs"]
TEXT_COLS = META_COLS + ["corrected_text"]

# every shard already touched by pass 1 or pass 2 -- pilot uses fresh ones
USED_SHARDS = {
    # pass 1 (stratified 200-clip audit)
    "filtered-train-00000-of-00082.parquet",
    "filtered-train-00001-of-00082.parquet",
    "filtered-train-00002-of-00082.parquet",
    "filtered-train-00003-of-00082.parquet",
    "filtered-train-00024-of-00082.parquet",
    "filtered-train-00045-of-00082.parquet",
    "filtered-train-00066-of-00082.parquet",
    # pass 2 (random truncation check)
    "filtered-train-00034-of-00082.parquet",
    "filtered-train-00039-of-00082.parquet",
    "filtered-train-00049-of-00082.parquet",
    "filtered-train-00053-of-00082.parquet",
    "filtered-train-00063-of-00082.parquet",
    "filtered-train-00075-of-00082.parquet",
    "filtered-train-00080-of-00082.parquet",
    "filtered-train-00086-of-00082.parquet",
}

WAV_RE = re.compile(r"^pl_(\d{5})_(filtered-train-\d{5}-of-\d{5})_r(\d+)\.wav$")


def hf_download_shard(shard: str, attempts: int = 8) -> Path:
    """Resumable download of one shard via the hf CLI; retries network drops."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(HF_EXE), "download", DATASET_ID,
        "--repo-type", "dataset",
        "--include", f"data/{shard}",
        "--local-dir", str(SHARD_DIR),
    ]
    for i in range(1, attempts + 1):
        print(f"[download] {shard}  (attempt {i}/{attempts})", flush=True)
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return SHARD_DIR / "data" / shard
        tqdm.write(f"  [!] {shard}: download failed (rc={result.returncode}), "
                   f"retrying in 10s -- partial file is resumed automatically")
        time.sleep(10)
    raise RuntimeError(f"could not download {shard} after {attempts} attempts")


def scan_existing() -> dict[str, dict[int, str]]:
    """shard -> {row_index: wav_name} for WAVs already extracted."""
    found: dict[str, dict[int, str]] = {}
    if not WAVS.exists():
        return found
    for p in sorted(WAVS.glob("*.wav")):
        m = WAV_RE.match(p.name)
        if m:
            found.setdefault(m.group(2) + ".parquet", {})[int(m.group(3))] = p.name
    return found


def main() -> None:
    meta = pd.read_csv(SHARD_META_CSV).sort_values("shard").reset_index(drop=True)
    candidates = meta[~meta["shard"].isin(USED_SHARDS)]["shard"].tolist()
    rng = random.Random(SEED)
    picked = sorted(rng.sample(candidates, N_SHARDS))
    print(f"[pick] {N_SHARDS} uniform-random shards (excluding all 15 audit shards):")
    for s in picked:
        print(f"   {s}")

    # ---- phase 1: metadata-only remote reads, row selection -------------
    plan: dict[str, list[int]] = {}
    fs = fsspec.filesystem("hf")
    for shard in tqdm(picked, desc="select rows"):
        with fs.open(f"hf://datasets/{DATASET_ID}/data/{shard}", "rb") as f:
            df = pq.read_table(f, columns=TEXT_COLS).to_pandas()
        band = df.index[df["duration_ss"] >= MIN_DUR].tolist()
        # per-shard quota: share the hour target evenly, sized by band mean
        mean_dur = float(df.loc[band, "duration_ss"].mean())
        quota = math.ceil((TARGET_HOURS * 3600 / N_SHARDS) / mean_dur)
        if len(band) < quota:
            tqdm.write(f"  [!] {shard}: only {len(band)} rows >= {MIN_DUR}s, taking all")
            rows = band
        else:
            rows = rng.sample(band, quota)
        plan[shard] = sorted(rows)
        df.loc[rows].to_pickle(MANIFESTS / f"_meta_{shard}.pkl")  # manifest cache

    total = sum(len(v) for v in plan.values())
    print(f"[plan] {total} clips (~{TARGET_HOURS:.0f} h target) across {len(plan)} shards")

    # ---- phase 2: download + extract, exempting existing WAVs -----------
    existing = scan_existing()
    WAVS.mkdir(parents=True, exist_ok=True)
    records = []
    all_existing_names = {n for v in existing.values() for n in v.values()}
    name_idx = 0
    while any(n.startswith(f"pl_{name_idx:05d}_") for n in all_existing_names):
        name_idx += 1
    for shard in tqdm(picked, desc="shards"):
        shard_rows = existing.get(shard, {})
        todo = [r for r in plan[shard] if r not in shard_rows]
        local = SHARD_DIR / "data" / shard
        df_full = None
        if todo:
            if not local.exists() or local.stat().st_size < 100_000_000:
                hf_download_shard(shard)
            table = pq.read_table(local, columns=TEXT_COLS + ["audio"])
            df_full = table.to_pandas()

        meta_df = pd.read_pickle(MANIFESTS / f"_meta_{shard}.pkl")
        for ridx in plan[shard]:
            stem = shard.replace(".parquet", "")
            wav_name = shard_rows.get(ridx)
            if wav_name is None:
                while any(n.startswith(f"pl_{name_idx:05d}_") for n in all_existing_names):
                    name_idx += 1
                wav_name = f"pl_{name_idx:05d}_{stem}_r{ridx}.wav"
                all_existing_names.add(wav_name)
                name_idx += 1
            if ridx not in shard_rows:
                row = df_full.loc[ridx]
                data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                sf.write(WAVS / wav_name, data, sr, subtype="PCM_16")
            mrow = meta_df.loc[ridx]
            records.append({
                "wav_file": wav_name,
                "text": mrow["corrected_text"],
                "duration": float(mrow["duration_ss"]),
                "mean_speech_prob": float(mrow["mean_speech_prob"]),
                "dbfs": float(mrow["dbfs"]),
                "shard": shard,
                "row_index": ridx,
            })

        if df_full is not None:
            del df_full
            local.unlink()  # free the 1.57 GB once extracted
            tqdm.write(f"[done] {shard} extracted, local shard removed")

    out = pd.DataFrame(records).reset_index(names="idx")
    out.to_csv(PILOT_CSV, index=False)

    # F5-TTS fine-tune manifest: audio|text|duration (relative to dataset dir)
    with open(PILOT_F5, "w", encoding="utf-8") as f:
        for r in records:
            text = r["text"].replace("|", " ").strip()  # pipe is the delimiter
            f.write(f"wav/{r['wav_file']}|{text}|{r['duration']:.3f}\n")

    for shard in picked:  # clean up temp metadata pickles
        (MANIFESTS / f"_meta_{shard}.pkl").unlink(missing_ok=True)

    hours = out["duration"].sum() / 3600
    print(f"\n[finished] {len(out)} clips, {hours:.1f} h -> {PILOT_CSV}")
    print(out["duration"].describe().to_string())


if __name__ == "__main__":
    sys.exit(main())
