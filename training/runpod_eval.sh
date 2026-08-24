#!/bin/bash
# ============================================================================
# Generate the 15 Ghanaian-English eval sentences from a fine-tuned
# checkpoint, using a trimmed pilot clip as the zero-shot voice prompt.
#
# All pilot clips are 12-15s; F5-TTS clips ref audio to 12s internally.
# We trim to 10s + match the transcript so the model generates target text
# instead of repeating the reference.
#
# Usage:  bash runpod_eval.sh <step> [ref_wav] [speed]
#   e.g.  bash runpod_eval.sh last
#         bash runpod_eval.sh last /workspace/ghana_pilot/wav/pl_00100_....wav 0.30
#
# Outputs land in /workspace/gen/step_<step>_speed<speed>/
# ============================================================================
set -e
STEP=${1:?usage: runpod_eval.sh <step> [ref_wav] [speed]}
SPEED=${3:-0.30}
cd /workspace/F5-TTS

SRC=ckpts/ghana_pilot/model_${STEP}.safetensors
if [ ! -f "$SRC" ]; then
    PT=ckpts/ghana_pilot/model_${STEP}.pt
    if [ -f "$PT" ]; then
        echo "safetensors not found, converting from .pt checkpoint..."
        bash /workspace/Ghana-tts/training/runpod_convert_ckpt.sh "$STEP"
    else
        echo "checkpoint not found: $SRC or $PT -- available:"
        ls ckpts/ghana_pilot/ 2>/dev/null || echo "(none)"
        exit 1
    fi
fi

VOCAB=data/ghana_pilot/vocab.txt
if [ ! -f "$VOCAB" ]; then
    echo "vocab not found: $VOCAB"
    exit 1
fi

# reference voice: given wav or the first pilot clip
REF_WAV=${2:-$(ls /workspace/ghana_pilot/wav/*.wav | head -1)}
CLIP_NAME=$(basename "$REF_WAV" .wav)
FULL_TEXT=$(grep -m1 "$CLIP_NAME" /workspace/ghana_pilot/pilot_f5.txt | cut -d'|' -f2)

# Trim reference audio to 10s and estimate matching text
TRIMMED_WAV=/workspace/ghana_pilot/ref_trimmed_${CLIP_NAME}.wav
python -c "
from pydub import AudioSegment
import soundfile as sf

clip = '${REF_WAV}'
full_text = '''${FULL_TEXT}'''

# Trim audio to 10 seconds
audio = AudioSegment.from_wav(clip)
trimmed = audio[:10000]
trimmed.export('${TRIMMED_WAV}', format='wav')

# Estimate text for 10s based on duration ratio
info = sf.info(clip)
ratio = 10.0 / info.duration
words = full_text.split()
n_words = max(5, int(len(words) * ratio))
ref_text = ' '.join(words[:n_words])
print(ref_text)
" 2>/dev/null > /tmp/ref_text.txt
REF_TEXT=$(cat /tmp/ref_text.txt)

echo "ref audio: $REF_WAV -> trimmed to 10s: $TRIMMED_WAV"
echo "ref text : ${REF_TEXT:0:80}..."
echo "ckpt     : $SRC"
echo "vocab    : $VOCAB"
echo "speed    : $SPEED"

SPEED_TAG=$(echo "$SPEED" | sed 's/\.//g')
OUT=/workspace/gen/step_${STEP}_speed${SPEED_TAG}_$(basename "$TRIMMED_WAV" .wav)
mkdir -p "$OUT"
i=0
# skip the 2 header lines, strip the "N. " numbering
tail -n +3 /workspace/ghana_pilot/test_sentences.txt | sed 's/^[0-9]*\. //' | while IFS= read -r sentence; do
    [ -z "$sentence" ] && continue
    i=$((i + 1))
    echo "[$i] $sentence"
    f5-tts_infer-cli \
        --model F5TTS_v1_Base \
        --ckpt_file "$SRC" \
        --vocab_file "$VOCAB" \
        --ref_audio "$TRIMMED_WAV" \
        --ref_text "$REF_TEXT" \
        --gen_text "$sentence" \
        --output_dir "$OUT" \
        --output_file "sent_$(printf '%02d' $i).wav" \
        --speed "$SPEED" || echo "  [!] generation failed for sentence $i"
done

echo
echo "done -- results in $OUT  (download the folder to listen)"
