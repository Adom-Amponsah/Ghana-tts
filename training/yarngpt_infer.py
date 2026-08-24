"""
YarnGPT inference — generate Ghanaian English speech from text.

No reference audio needed. Just text → speech.

Run on RunPod:
  python /workspace/Ghana-tts/training/yarngpt_infer.py \
    --model_path /workspace/yarngpt_ghana/final \
    --text "The President arrived in Accra late yesterday evening."
"""

import argparse
import os
import sys
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from outetts.wav_tokenizer.decoder import WavTokenizer

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, help="Path to trained model")
parser.add_argument("--text", required=True, help="Text to synthesize")
parser.add_argument("--output", default="/workspace/yarngpt_output.wav", help="Output WAV path")
parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
parser.add_argument("--max_length", type=int, default=4000, help="Max generation length")
parser.add_argument("--wavtok_config",
                    default="/workspace/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
parser.add_argument("--wavtok_model",
                    default="/workspace/wavtokenizer_large_speech_320_24k.ckpt")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load model + tokenizer
# ---------------------------------------------------------------------------
print(f"\nLoading model from {args.model_path}...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto").to(device)
model.eval()
print(f"  Vocab: {len(tokenizer)}")
print(f"  Params: {model.num_parameters():,}")

# ---------------------------------------------------------------------------
# Load WavTokenizer for decoding
# ---------------------------------------------------------------------------
print("Loading WavTokenizer...")
wavtokenizer = WavTokenizer.from_pretrained0802(args.wavtok_config, args.wavtok_model)
wavtokenizer = wavtokenizer.to(device)
print("  Done.")

# ---------------------------------------------------------------------------
# Build prompt
# ---------------------------------------------------------------------------
import re
import inflect
import uroman as ur

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
text_str = "<|text_sep|>".join(words)

# Prompt: text section + language + audio_start (model generates the rest)
prompt = (
    f"<|im_start|>\n"
    f"<|text_start|>{text_str}<|text_end|>\n"
    f"<|english|>\n"
    f"<|audio_start|>\n"
)

print(f"\nPrompt (first 300 chars): {prompt[:300]}...")
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
input_length = input_ids.shape[1]
print(f"  Prompt tokens: {input_length}")

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
print(f"\nGenerating (temperature={args.temperature}, max_length={args.max_length})...")

with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        do_sample=True,
    )

print(f"  Output tokens: {output.shape[1]}")
print(f"  Generated tokens: {output.shape[1] - input_length}")

# ---------------------------------------------------------------------------
# Decode audio codes from output
# ---------------------------------------------------------------------------
def extract_integers(s):
    import re as _re
    matches = _re.findall(r'\|(-?\d+)\|', s)
    return [int(m) for m in matches]

decoded = tokenizer.decode(output[0][input_length:])
codes = extract_integers(decoded)

print(f"  Extracted {len(codes)} audio codes")
if len(codes) == 0:
    print("  ERROR: No audio codes extracted. Model may not have learned to generate codes.")
    print(f"  Raw output (first 500 chars): {decoded[:500]}")
    sys.exit(1)

print(f"  Code range: {min(codes)} - {max(codes)}")

# ---------------------------------------------------------------------------
# Decode codes → audio
# ---------------------------------------------------------------------------
print("\nDecoding audio codes with WavTokenizer...")
discrete_code = torch.tensor([[codes]]).to(device)
features = wavtokenizer.codes_to_features(discrete_code).to(device)
bandwidth_id = torch.tensor([0]).to(device)
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
audio_out = audio_out.to("cpu")

print(f"  Audio shape: {audio_out.shape}")
print(f"  Duration: {audio_out.shape[-1] / 24000:.2f}s")

# Save
torchaudio.save(args.output, audio_out, sample_rate=24000)
print(f"\n  Saved: {args.output}")
print(f"\nDone! Listen with:  ffplay {args.output}")
