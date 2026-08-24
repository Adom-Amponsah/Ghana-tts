"""
YarnGPT pipeline test — prove the full chain works on ONE Ghanaian clip before
processing all 3,933.

Pipeline:
  Ghana WAV (16kHz)
    → resample to 24kHz
    → WavTokenizer encode → audio codes (list of ints)
    → build training string (text + codes with special tokens)
    → SmolLM2 tokenizer → input_ids
    → SmolLM2-360M forward pass → loss
    → verify shapes, loss is finite, no errors

Run on RunPod:
  cd /workspace
  python Ghana-tts/training/yarngpt_pipeline_test.py \
    --wav /workspace/ghana_pilot/wav/pl_00000_filtered-train-00017-of-00082_r0.wav \
    --text "time the government decided not to do a renewal and to nationalize"
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--wav", required=True, help="Path to a Ghanaian pilot WAV")
parser.add_argument("--text", required=True, help="Transcript for that WAV")
parser.add_argument("--smollm", default="HuggingFaceTB/SmolLM2-360M",
                    help="SmolLM2 checkpoint (HF id or local path)")
parser.add_argument("--wavtok_config",
                    default="/workspace/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                    help="WavTokenizer config YAML")
parser.add_argument("--wavtok_model",
                    default="/workspace/wavtokenizer_large_speech_320_24k.ckpt",
                    help="WavTokenizer checkpoint")
args = parser.parse_args()

print("=" * 70)
print("YarnGPT Pipeline Test — one clip end-to-end")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Install / import dependencies
# ---------------------------------------------------------------------------
print("\n[1/8] Importing dependencies...")

try:
    import torch
    import torchaudio
    import numpy as np
    import inflect
    import uroman as ur
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from outetts.wav_tokenizer.decoder import WavTokenizer
    from outetts.wav_tokenizer.encoder.utils import convert_audio
except ImportError as e:
    print(f"  Missing dependency: {e}")
    print("  Install with:  pip install outetts uroman inflect torchaudio")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# 2. Load SmolLM2-360M + tokenizer
# ---------------------------------------------------------------------------
print("\n[2/8] Loading SmolLM2-360M tokenizer + model...")

tokenizer = AutoTokenizer.from_pretrained(args.smollm)
model = AutoModelForCausalLM.from_pretrained(args.smollm, torch_dtype="auto").to(device)

print(f"  Base vocab size: {len(tokenizer)}")
print(f"  Model params: {model.num_parameters():,}")
print(f"  Model memory: {model.get_memory_footprint() / 1e6:.1f} MB")

# ---------------------------------------------------------------------------
# 3. Add special tokens (exactly as YarnGPT does)
# ---------------------------------------------------------------------------
print("\n[3/8] Adding special tokens...")

special_tokens = [
    "<|im_start|>",
    "<|im_end|>",
    "<|text_start|>",
    "<|text_end|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|code_start|>",
    "<|code_end|>",
    "<|text_sep|>",
]
n_special = tokenizer.add_tokens(special_tokens)
print(f"  Added {n_special} special tokens")

# Audio code tokens: <|0|> through <|2023|>
audio_tokens = [f"<|{i}|>" for i in range(0, 2024)]
n_audio = tokenizer.add_tokens(audio_tokens)
print(f"  Added {n_audio} audio code tokens (<|0|> .. <|2023|>)")

# Time tokens: <|t_0.00|> through <|t_9.99|>
time_tokens = [f"<|t_{round(i,2)}|>" for i in np.arange(0, 10, 0.01)]
n_time = tokenizer.add_tokens(time_tokens)
print(f"  Added {n_time} time tokens")

# Language tokens
lang_tokens = tokenizer.add_tokens(["<|english|>", "<|hausa|>", "<|igbo|>", "<|yoruba|>"])
print(f"  Added {lang_tokens} language tokens")

# Pad token
tokenizer.pad_token_id = 0

# Resize model embeddings
total_tokens = 49152 + 4096  # SmolLM2 base (49152) + extended vocab
print(f"  Total target vocab: {total_tokens}")
print(f"  Current tokenizer len: {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))
print(f"  Resized embeddings to: {model.config.vocab_size}")

# ---------------------------------------------------------------------------
# 4. Load WavTokenizer
# ---------------------------------------------------------------------------
print("\n[4/8] Loading WavTokenizer...")

if not os.path.exists(args.wavtok_config):
    print(f"  WavTokenizer config not found: {args.wavtok_config}")
    print("  Download with:")
    print("  wget https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml -O /workspace/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
    sys.exit(1)

if not os.path.exists(args.wavtok_model):
    print(f"  WavTokenizer model not found: {args.wavtok_model}")
    print("  Download with:")
    print("  gdown 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt -O /workspace/wavtokenizer_large_speech_320_24k.ckpt")
    sys.exit(1)

wavtokenizer = WavTokenizer.from_pretrained0802(args.wavtok_config, args.wavtok_model)
wavtokenizer = wavtokenizer.to(device)
print("  WavTokenizer loaded")

# ---------------------------------------------------------------------------
# 5. Load + resample audio
# ---------------------------------------------------------------------------
print("\n[5/8] Loading and resampling audio...")

audio_data, sample_rate = torchaudio.load(args.wav)
audio_data = audio_data.squeeze()
print(f"  Original: {audio_data.shape}, sr={sample_rate}")

# Resample to 24kHz
audio_f32 = audio_data.to(dtype=torch.float32).unsqueeze(0)
audio_24k = convert_audio(audio_f32, sample_rate, 24000, 1).to(device)
if audio_24k.ndim == 3:
    audio_24k = audio_24k.squeeze(1)
print(f"  Resampled: {audio_24k.shape}, sr=24000")
print(f"  Duration: {audio_24k.shape[-1] / 24000:.2f}s")

# ---------------------------------------------------------------------------
# 6. Encode with WavTokenizer → audio codes
# ---------------------------------------------------------------------------
print("\n[6/8] Encoding audio with WavTokenizer...")

bandwidth_id = torch.tensor([0]).to(device)
with torch.no_grad():
    _, codes = wavtokenizer.encode_infer(audio_24k, bandwidth_id=bandwidth_id)
codes = codes.squeeze(1).to(device)
code_list = codes[0].tolist()
print(f"  Codes shape: {codes.shape}")
print(f"  Number of audio codes: {len(code_list)}")
print(f"  Code range: {min(code_list)} - {max(code_list)}")
print(f"  First 10 codes: {code_list[:10]}")

# Verify codes are within token range
assert max(code_list) < 2024, f"Code {max(code_list)} exceeds vocab range (0-2023)"
assert min(code_list) >= 0, f"Negative code {min(code_list)}"

# ---------------------------------------------------------------------------
# 7. Build training string
# ---------------------------------------------------------------------------
print("\n[7/8] Building YarnGPT training string...")

# Process text: lowercase, number-to-words, clean
import re
lec = inflect.engine()
uroman = ur.Uroman()

def process_text(text):
    text = uroman.romanize_string(text)
    text = re.sub(r'\d+(\.\d+)?', lambda x: lec.number_to_words(x.group()), text.lower())
    text = re.sub(r'[-_/,\.\\]', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

words = process_text(args.text)
print(f"  Processed text: {words}")

# Build the training string
# Format (from notebook + AudioTokenizerV2):
# <|im_start|>\n<|text_start|>word1<|text_sep|>word2<|text_end|>\n<|english|>\n<|audio_start|>\n<|code_start|><|123|><|456|>...<|code_end|><|audio_end|>\n<|im_end|>
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

print(f"\n  Training string (first 500 chars):\n  {training_string[:500]}...")
print(f"\n  Training string length: {len(training_string)} chars")

# Tokenize the training string
input_ids = tokenizer.encode(training_string, add_special_tokens=False, return_tensors="pt").to(device)
print(f"  Tokenized: {input_ids.shape}")
print(f"  Input IDs (first 20): {input_ids[0][:20].tolist()}")
print(f"  Input IDs (last 20): {input_ids[0][-20:].tolist()}")

# Verify all token IDs are within vocab range
assert input_ids.max().item() < len(tokenizer), f"Token ID {input_ids.max().item()} >= vocab size {len(tokenizer)}"
assert input_ids.min().item() >= 0, f"Negative token ID {input_ids.min().item()}"
print(f"  All token IDs valid (0 to {len(tokenizer) - 1})")

# ---------------------------------------------------------------------------
# 8. Forward pass through model
# ---------------------------------------------------------------------------
print("\n[8/8] Running forward pass through SmolLM2-360M...")

model.train()
labels = input_ids.clone()

t0 = time.time()
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(input_ids=input_ids, labels=labels)
elapsed = time.time() - t0

loss = outputs.loss
logits = outputs.logits

print(f"\n  Forward pass completed in {elapsed:.2f}s")
print(f"  Loss: {loss.item():.4f}")
print(f"  Logits shape: {logits.shape}")
print(f"  Loss is finite: {torch.isfinite(loss).item()}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PIPELINE TEST RESULT")
print("=" * 70)

checks = [
    ("SmolLM2 loaded", model is not None),
    ("Special tokens added", len(tokenizer) > 49152),
    ("WavTokenizer loaded", wavtokenizer is not None),
    ("Audio resampled to 24kHz", audio_24k.shape[-1] > 0),
    ("Audio codes generated", len(code_list) > 0),
    ("Codes in valid range", max(code_list) < 2024),
    ("Training string built", len(training_string) > 0),
    ("Tokenization succeeded", input_ids.shape[1] > 0),
    ("Token IDs valid", input_ids.max().item() < len(tokenizer)),
    ("Forward pass succeeded", loss is not None),
    ("Loss is finite", torch.isfinite(loss).item()),
]

all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {name}")

print()
if all_pass:
    print("ALL CHECKS PASSED — pipeline is ready to scale to 3,933 clips.")
    print("\nNext steps:")
    print("  1. Run data prep on all clips → build training CSV")
    print("  2. Launch training (5 epochs, batch_size=4, lr=1e-3)")
    print("  3. Test inference with Ghanaian English sentences")
else:
    print("SOME CHECKS FAILED — fix before scaling.")
    sys.exit(1)
