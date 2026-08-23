"""Pass 2: genuinely-random 12-15 s truncation check sample.

Closes the representativeness gap of the 200-clip audit: uniform-random
shard selection (excluding the 7 shards already sampled), uniform-random
rows from the 12-15 s band, 8 shards x 6 clips = 48 clips.

Outputs:
  data/samples_pass2/*.wav           named p2_{idx:02d}_{shardstem}_r{row}.wav
  data/manifests/sample_pass2.csv    manifest incl. at_ceiling flag

Phase 1 (remote metadata reads, seconds) selects rows; phase 2 downloads
only the 8 shards (resumable hf CLI), extracts the picked clips, then frees
each 1.57 GB shard file. Re-runnable: existing WAVs are exempted.
"""

import io
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
SAMPLES = ROOT / "data" / "samples_pass2"
SHARD_DIR = ROOT / "data" / "shards_pass2"
SHARD_META_CSV = MANIFESTS / "shard_metadata.csv"
SAMPLE_CSV = MANIFESTS / "sample_pass2.csv"
HF_EXE = ROOT / ".venv" / "Scripts" / "hf.exe"

SEED = 777                    # different seed family from pass 1 (42)
N_SHARDS = 8
N_PER_SHARD = 6
DUR_LO, DUR_HI = 12.0, 15.0   # the band that survives the >=12 s floor
CEILING = 14.8                # ceiling-targeted threshold (only 4.3% of band >=14.9)
N_CEILING_SHARDS = 4          # add-on clips from 4 shards only (3 each = 12)
META_COLS = ["duration_ss", "mean_speech_prob", "dbfs"]
TEXT_COLS = META_COLS + ["corrected_text"]

USED_SHARDS_PASS1 = {
    "filtered-train-00000-of-00082.parquet",
    "filtered-train-00001-of-00082.parquet",
    "filtered-train-00002-of-00082.parquet",
    "filtered-train-00003-of-00082.parquet",
    "filtered-train-00024-of-00082.parquet",
    "filtered-train-00045-of-00082.parquet",
    "filtered-train-00066-of-00082.parquet",
}

WAV_RE = re.compile(r"^p2_(\d{2})_(filtered-train-\d{5}-of-\d{5})_r(\d+)\.wav$")


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
    if not SAMPLES.exists():
        return found
    for p in sorted(SAMPLES.glob("*.wav")):
        m = WAV_RE.match(p.name)
        if m:
            found.setdefault(m.group(2) + ".parquet", {})[int(m.group(3))] = p.name
    return found


def main() -> None:
    meta = pd.read_csv(SHARD_META_CSV).sort_values("shard").reset_index(drop=True)
    candidates = meta[~meta["shard"].isin(USED_SHARDS_PASS1)]["shard"].tolist()
    rng = random.Random(SEED)
    picked = sorted(rng.sample(candidates, N_SHARDS))
    print(f"[pick] {N_SHARDS} uniform-random shards (excluding pass-1 shards):")
    for s in picked:
        print(f"   {s}")

    # ---- phase 1: metadata-only remote reads, row selection -------------
    plan: dict[str, list[int]] = {}
    ceiling_plan: dict[str, list[int]] = {}
    fs = fsspec.filesystem("hf")
    for shard in tqdm(picked, desc="select rows"):
        with fs.open(f"hf://datasets/{DATASET_ID}/data/{shard}", "rb") as f:
            df = pq.read_table(f, columns=TEXT_COLS).to_pandas()
        band = df.index[df["duration_ss"].between(DUR_LO, DUR_HI)].tolist()
        if len(band) < N_PER_SHARD:
            tqdm.write(f"  [!] {shard}: only {len(band)} rows in band, taking all")
            rows = band
        else:
            rows = rng.sample(band, N_PER_SHARD)
        plan[shard] = sorted(rows)
        # ceiling-targeted add-on: uniform random is starved here (only 4.3%
        # of the band sits >= 14.9 s), so stratify 3 per shard -- but only for
        # 4 of the 8 shards to halve the re-download cost (6.3 GB, not 12.5)
        if shard in picked[: N_CEILING_SHARDS]:
            near_cap = [i for i in df.index
                        if df.at[i, "duration_ss"] >= CEILING and i not in plan[shard]]
            ceiling_plan[shard] = sorted(rng.sample(near_cap, min(len(near_cap), 3)))
        else:
            ceiling_plan[shard] = []
        # cache metadata for the manifest (avoid re-reading after download)
        keep = sorted(set(plan[shard]) | set(ceiling_plan[shard]))
        df.loc[keep].to_pickle(MANIFESTS / f"_meta_{shard}.pkl")

    total = sum(len(v) for v in plan.values())
    n_ceiling = sum(len(v) for v in ceiling_plan.values())
    print(f"[plan] {total} uniform-random clips + {n_ceiling} ceiling-targeted "
          f"across {len(plan)} shards")

    # ---- phase 2: download + extract, exempting existing WAVs -----------
    existing = scan_existing()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    records = []
    idx = 0
    # free-running name counter so new WAVs never collide with existing ones
    all_existing_names = {n for v in existing.values() for n in v.values()}
    name_idx = 0
    while any(n.startswith(f"p2_{name_idx:02d}_") for n in all_existing_names):
        name_idx += 1
    for shard in tqdm(picked, desc="shards"):
        shard_rows = existing.get(shard, {})
        needed = set(plan[shard]) | set(ceiling_plan[shard])
        todo = [r for r in needed if r not in shard_rows]
        local = SHARD_DIR / "data" / shard
        df_full = None
        if todo:
            if not local.exists() or local.stat().st_size < 100_000_000:
                hf_download_shard(shard)
            table = pq.read_table(local, columns=TEXT_COLS + ["audio"])
            df_full = table.to_pandas()

        meta_df = pd.read_pickle(MANIFESTS / f"_meta_{shard}.pkl")
        for ridx in sorted(set(plan[shard]) | set(ceiling_plan[shard])):
            stem = shard.replace(".parquet", "")
            wav_name = shard_rows.get(ridx)
            if wav_name is None:
                while any(n.startswith(f"p2_{name_idx:02d}_") for n in all_existing_names):
                    name_idx += 1
                wav_name = f"p2_{name_idx:02d}_{stem}_r{ridx}.wav"
                all_existing_names.add(wav_name)
                name_idx += 1
            if ridx not in shard_rows:
                row = df_full.loc[ridx]
                data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                sf.write(SAMPLES / wav_name, data, sr, subtype="PCM_16")
            mrow = meta_df.loc[ridx]
            records.append({
                "pass2_index": idx,
                "shard": shard,
                "row_index": ridx,
                "wav_file": wav_name,
                "selection": "ceiling_target" if ridx in ceiling_plan[shard] else "uniform_random",
                "duration_ss": float(mrow["duration_ss"]),
                "mean_speech_prob": float(mrow["mean_speech_prob"]),
                "dbfs": float(mrow["dbfs"]),
                "at_ceiling": bool(mrow["duration_ss"] >= CEILING),
                "corrected_text": mrow["corrected_text"],
            })
            idx += 1

        if df_full is not None:
            del df_full
            local.unlink()  # free the 1.57 GB once extracted
            tqdm.write(f"[done] {shard} extracted, local shard removed")

    out = pd.DataFrame(records)
    out.to_csv(SAMPLE_CSV, index=False)
    for shard in picked:  # clean up temp metadata pickles
        (MANIFESTS / f"_meta_{shard}.pkl").unlink(missing_ok=True)

    print(f"\n[finished] {len(out)} clips -> {SAMPLE_CSV}")
    print(f"at_ceiling (>= {CEILING}s): {int(out['at_ceiling'].sum())}/{len(out)}")
    print(out[["duration_ss", "mean_speech_prob", "dbfs"]].describe().to_string())


if __name__ == "__main__":
    sys.exit(main())
