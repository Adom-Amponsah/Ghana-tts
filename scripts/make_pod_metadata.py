"""Build pod-side metadata.csv (absolute /workspace paths) and refresh the
pilot bundle zip with it. The pod's prepare_csv_wavs.py expects:
    audio_file|text          (header, pipe-delimited, absolute paths)
"""

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
PILOT_CSV = ROOT / "data" / "manifests" / "pilot.csv"
BUNDLE = ROOT / "data" / "pilot_bundle.zip"
META_OUT = TRAINING / "metadata.csv"

POD_WAV_DIR = "/workspace/ghana_pilot/wav"


def main() -> None:
    TRAINING.mkdir(exist_ok=True)
    df = pd.read_csv(PILOT_CSV)
    lines = ["audio_file|text"]
    for r in df.itertuples():
        text = str(r.text).replace("|", " ").replace("\n", " ").strip()
        lines.append(f"{POD_WAV_DIR}/{r.wav_file}|{text}")
    META_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {META_OUT.name}: {len(lines) - 1} rows")

    with zipfile.ZipFile(BUNDLE, "a") as z:
        z.write(META_OUT, "metadata.csv")
    print(f"bundle size: {BUNDLE.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
