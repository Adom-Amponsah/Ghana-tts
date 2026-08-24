"""
YarnGPT data prep — process all pilot clips into YarnGPT training format.

Input:  /workspace/ghana_pilot/wav/*.wav + metadata.csv
Output: /workspace/yarngpt_train_data.csv  (column "0" = training string)

Each row is a string like:
  <|im_start|>\n<|text_start|>word1<|text_sep|>word2<|text_end|>\n<|english|>\n
  <|audio_start|>\n<|code_start|><|123|><|456|>...<|code_end|><|audio_end|>\n<|im_end|>

Run on RunPod:
  cd /workspace
  python Ghana-tts/training/yarngpt_data_prep.py
"""

import os
import re
import sys
import time
import inflect
import uroman as ur
import numpy as np
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from outetts.wav_tokenizer.decoder import WavTokenizer
from outetts.wav_tokenizer.encoder.utils import convert_audio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WAV_DIR = Path("/workspace/ghana_pilot/wav")
METADATA_CSV = Path("/workspace/ghana_pilot/metadata.csv")
OUTPUT_CSV = Path("/workspace/yarngpt_train_data.csv")

SMOLLM_PATH = "HuggingFaceTB/SmolLM2-360M"
WAVTOK_CONFIG = "/workspace/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_MODEL = "/workspace/wavtokenizer_large_speech_320_24k.ckpt"

# Text processing (same as YarnGPT's AudioTokenizerV2)
lec = inflect.engine()
uroman = ur.Uroman()

def process_text(text: str):
    text = uroman.romanize_string(text)
    text = re.sub(r'\d+(\.\d+)?', lambda x: lec.number_to_words(x.group()), text.lower())
    text = re.sub(r'[-_/,\.\\]', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load WavTokenizer
print("Loading WavTokenizer...")
wavtokenizer = WavTokenizer.from_pretrained0802(WAVTOK_CONFIG, WAVTOK_MODEL)
wavtokenizer = wavtokenizer.to(device)
print("  Done.")

# Load metadata
print(f"Loading metadata from {METADATA_CSV}...")
meta = pd.read_csv(METADATA_CSV)
print(f"  {len(meta)} clips in metadata")

# Check which WAVs exist
meta["wav_path"] = meta["wav_file"].apply(lambda f: WAV_DIR / f)
meta = meta[meta["wav_path"].apply(lambda p: p.exists())].reset_index(drop=True)
print(f"  {len(meta)} clips have WAV files on disk")

# ---------------------------------------------------------------------------
# Process clips
# ---------------------------------------------------------------------------
results = []
errors = []
bandwidth_id = torch.tensor([0]).to(device)

print(f"\nProcessing {len(meta)} clips...")
t0 = time.time()

for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Tokenizing"):
    wav_path = str(row["wav_path"])
    text = str(row["corrected_text"])

    try:
        # Load audio
        audio_data, sr = torchaudio.load(wav_path)
        audio_data = audio_data.squeeze()

        # Skip if too short or corrupt
        if audio_data.numel() < sr * 1.0:
            errors.append((row["wav_file"], "too_short"))
            continue

        # Resample to 24kHz
        audio_f32 = audio_data.to(dtype=torch.float32).unsqueeze(0)
        audio_24k = convert_audio(audio_f32, sr, 24000, 1).to(device)
        if audio_24k.ndim == 3:
            audio_24k = audio_24k.squeeze(1)

        # Encode with WavTokenizer
        with torch.no_grad():
            _, codes = wavtokenizer.encode_infer(audio_24k, bandwidth_id=bandwidth_id)
        codes = codes.squeeze(1).to(device)
        code_list = codes[0].tolist()

        # Skip if no codes
        if len(code_list) == 0:
            errors.append((row["wav_file"], "no_codes"))
            continue

        # Process text
        words = process_text(text)
        if len(words) == 0:
            errors.append((row["wav_file"], "empty_text"))
            continue

        # Build training string
        text_str = "<|text_sep|>".join(words)
        codes_str = "".join([f"<|{c}|>" for c in code_list])

        training_string = (
            f"<|im_start|>\n"
            f"<|text_start|>{text_str}<|text_end|>\n"
            f"<|english|>\n"
            f"<|audio_start|>\n"
            f"<|code_start|>{codes_str}<|code_end|><|audio_end|>\n"
            f"<|im_end|>"
        )

        results.append({
            "0": training_string,
            "length": len(training_string),
            "wav_file": row["wav_file"],
            "n_codes": len(code_list),
            "n_words": len(words),
            "duration_ss": row.get("duration_ss", 0),
        })

    except Exception as e:
        errors.append((row["wav_file"], str(e)[:100]))
        continue

elapsed = time.time() - t0

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"DATA PREP COMPLETE")
print(f"{'=' * 60}")
print(f"  Processed:  {len(results)} clips")
print(f"  Errors:     {len(errors)} clips")
print(f"  Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Speed:      {len(results)/elapsed:.1f} clips/s")

if errors:
    print(f"\n  Error breakdown:")
    from collections import Counter
    reasons = Counter(e[1] for e in errors)
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count}")
    # Save error log
    err_df = pd.DataFrame(errors, columns=["wav_file", "reason"])
    err_df.to_csv("/workspace/yarngpt_data_prep_errors.csv", index=False)
    print(f"  Error log: /workspace/yarngpt_data_prep_errors.csv")

# Save training CSV (only the "0" column, same format as YarnGPT notebook)
df = pd.DataFrame(results)
df[["0"]].to_csv(OUTPUT_CSV, index=False)
print(f"\n  Training CSV: {OUTPUT_CSV} ({len(df)} rows)")

# Stats
print(f"\n  Training string lengths:")
print(f"    min:    {df['length'].min()}")
print(f"    max:    {df['length'].max()}")
print(f"    mean:   {df['length'].mean():.0f}")
print(f"    median: {df['length'].median():.0f}")

print(f"\n  Audio codes per clip:")
print(f"    min:    {df['n_codes'].min()}")
print(f"    max:    {df['n_codes'].max()}")
print(f"    mean:   {df['n_codes'].mean():.0f}")

print(f"\n  Total audio codes: {df['n_codes'].sum():,}")
print(f"  Total training tokens (approx): {df['length'].sum():,}")

print(f"\nReady for training. Next:")
print(f"  python /workspace/Ghana-tts/training/yarngpt_train.py")
