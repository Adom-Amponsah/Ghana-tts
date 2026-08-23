"""Automated audio QC pass over the 200-clip sample.

Computes per-clip DSP features and heuristic flags so the human listening
session can focus on suspicious clips. No network access; local WAVs only.

Features per clip:
  clip_frac        fraction of samples at digital full-scale (>= 0.999)
  silence_frac     fraction of 20 ms frames below -50 dBFS
  duty             fraction of frames within 12 dB of the clip's own median
  mod_std          std of frame-level dBFS (temporal modulation; speech is
                   high, steady music/noise is low)
  flatness         mean spectral flatness (high -> noise-like/music)
  centroid         mean spectral centroid
  start_ratio      RMS(first 150 ms) / RMS(whole)   (>1: starts hot, no lead-in)
  end_ratio        RMS(last 150 ms) / RMS(whole)    (>1: ends hot, likely cut)

Flags (heuristics, to be confirmed by ear):
  clipping         clip_frac > 0.0005
  mostly_silent    silence_frac > 0.5
  possible_music   flatness > 0.06 and mod_std < 7
  abrupt_end       end_ratio > 1.25
  abrupt_start     start_ratio > 1.5
  low_activity     duty < 0.45

Output: analysis/clip_qc.csv  (+ printed summary)
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
MANIFEST = ROOT / "data" / "manifests" / "sample_200.csv"
OUT = ROOT / "analysis" / "clip_qc.csv"

FRAME = 320   # 20 ms at 16 kHz
HOP = 160


def analyze(x: np.ndarray, sr: int) -> dict:
    rms = librosa.feature.rms(y=x, frame_length=FRAME, hop_length=HOP)[0]
    rms = np.maximum(rms, 1e-10)
    db = 20.0 * np.log10(rms)

    edge = max(1, int(0.15 * sr))
    whole_rms = float(np.sqrt(np.mean(x ** 2))) + 1e-10
    start_rms = float(np.sqrt(np.mean(x[:edge] ** 2)))
    end_rms = float(np.sqrt(np.mean(x[-edge:] ** 2)))

    flat = librosa.feature.spectral_flatness(y=x, hop_length=512)[0]
    cent = librosa.feature.spectral_centroid(y=x, sr=sr, hop_length=512)[0]

    return {
        "clip_frac": float(np.mean(np.abs(x) >= 0.999)),
        "silence_frac": float(np.mean(db < -50.0)),
        "duty": float(np.mean(db > np.median(db) - 12.0)),
        "mod_std": float(np.std(db)),
        "flatness": float(np.mean(flat)),
        "centroid": float(np.mean(cent)),
        "start_ratio": start_rms / whole_rms,
        "end_ratio": end_rms / whole_rms,
    }


def flag_row(r: pd.Series) -> list[str]:
    flags = []
    if r["clip_frac"] > 0.0005:
        flags.append("clipping")
    if r["silence_frac"] > 0.5:
        flags.append("mostly_silent")
    if r["flatness"] > 0.06 and r["mod_std"] < 7.0:
        flags.append("possible_music")
    if r["end_ratio"] > 1.25:
        flags.append("abrupt_end")
    if r["start_ratio"] > 1.5:
        flags.append("abrupt_start")
    if r["duty"] < 0.45:
        flags.append("low_activity")
    return flags


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)

    rows = []
    for _, m in tqdm(manifest.iterrows(), total=len(manifest), desc="qc"):
        x, sr = sf.read(SAMPLES / m["wav_file"], dtype="float32")
        feats = analyze(x, sr)
        rows.append({"wav_file": m["wav_file"], "stratum": m["stratum"],
                     "shard": m["shard"], **feats})

    qc = pd.DataFrame(rows)
    qc["flags"] = qc.apply(flag_row, axis=1)
    qc["n_flags"] = qc["flags"].str.len()
    qc["flags"] = qc["flags"].apply(lambda f: ";".join(f))
    qc.to_csv(OUT, index=False)

    print(f"\n[done] {len(qc)} clips -> {OUT}")
    print("\nflag counts:")
    all_flags = [f for fs in qc["flags"] if fs for f in fs.split(";")]
    for name, n in pd.Series(all_flags).value_counts().items():
        print(f"  {name:16s} {n}")
    print(f"\nclips with >=1 flag: {(qc['n_flags'] > 0).sum()}/{len(qc)}")

    print("\nfeature summary:")
    cols = ["clip_frac", "silence_frac", "duty", "mod_std", "flatness", "start_ratio", "end_ratio"]
    print(qc[cols].describe().loc[["mean", "50%", "min", "max"]].round(4).to_string())

    print("\nflagged clips:")
    for _, r in qc[qc["n_flags"] > 0].iterrows():
        print(f"  {r['wav_file']}  [{r['flags']}]")


if __name__ == "__main__":
    sys.exit(main())
