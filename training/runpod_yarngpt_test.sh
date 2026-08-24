#!/bin/bash
# ============================================================================
# YarnGPT pipeline test — setup + run on RunPod
# Installs all deps, downloads WavTokenizer, runs the one-clip test.
# ============================================================================
set -e
cd /workspace

echo "========================================================"
echo "YarnGPT Pipeline Test — Setup"
echo "========================================================"

# ---- 1. Install dependencies -----------------------------------------------
echo "[1/4] Installing Python dependencies..."

pip install -q hf_transfer outetts==0.2.3 uroman inflect gdown 2>&1 | tail -5

# Unset hf_transfer env var if the package failed to install
if ! python -c "import hf_transfer" 2>/dev/null; then
    echo "  hf_transfer not available, disabling HF_HUB_ENABLE_HF_TRANSFER"
    export HF_HUB_ENABLE_HF_TRANSFER=0
fi

echo "  Done."

# ---- 2. Download WavTokenizer files ----------------------------------------
echo "[2/4] Downloading WavTokenizer config + checkpoint..."

WAVTOK_YAML="/workspace/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
WAVTOK_CKPT="/workspace/wavtokenizer_large_speech_320_24k.ckpt"

if [ ! -f "$WAVTOK_YAML" ]; then
    wget -q "https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/resolve/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml" \
        -O "$WAVTOK_YAML"
    echo "  Downloaded WavTokenizer config."
else
    echo "  WavTokenizer config already exists."
fi

if [ ! -f "$WAVTOK_CKPT" ]; then
    gdown 1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt -O "$WAVTOK_CKPT" 2>&1 | tail -3
    echo "  Downloaded WavTokenizer checkpoint."
else
    echo "  WavTokenizer checkpoint already exists."
fi

# ---- 3. Pull latest code ---------------------------------------------------
echo "[3/4] Pulling latest code..."
cd /workspace/Ghana-tts
git pull -q 2>/dev/null || true
echo "  Done."

# ---- 4. Run the pipeline test ----------------------------------------------
echo "[4/4] Running pipeline test..."
echo ""

# Find a pilot WAV if none specified
WAV_FILE=$(ls /workspace/ghana_pilot/wav/*.wav 2>/dev/null | head -1)
if [ -z "$WAV_FILE" ]; then
    echo "ERROR: No WAV files found in /workspace/ghana_pilot/wav/"
    echo "       Make sure the pilot dataset is unpacked."
    exit 1
fi

# Get the transcript from metadata.csv for this WAV
WAV_NAME=$(basename "$WAV_FILE")
TEXT=$(python -c "
import pandas as pd, sys
df = pd.read_csv('/workspace/ghana_pilot/metadata.csv')
row = df[df['wav_file']==sys.argv[1]]
if len(row): print(row.iloc[0]['corrected_text'])
else: print('the government decided to nationalize')
" "$WAV_NAME" 2>/dev/null || echo "the government decided to nationalize")

echo "  WAV:  $WAV_FILE"
echo "  Text: $TEXT"
echo ""

python /workspace/Ghana-tts/training/yarngpt_pipeline_test.py \
    --wav "$WAV_FILE" \
    --text "$TEXT" \
    --wavtok_config "$WAVTOK_YAML" \
    --wavtok_model "$WAVTOK_CKPT"
