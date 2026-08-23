#!/bin/bash
# ============================================================================
# Generate the 15 Ghanaian-English eval sentences from a fine-tuned
# checkpoint, using one pilot clip as the zero-shot voice prompt.
#
# Usage:  bash runpod_eval.sh <step> [ref_wav]
#   e.g.  bash runpod_eval.sh 15000
#         bash runpod_eval.sh 15000 /workspace/ghana_pilot/wav/pl_00100_....wav
#
# Outputs land in /workspace/gen/step_<step>/
# ============================================================================
set -e
STEP=${1:?usage: runpod_eval.sh <step> [ref_wav]}
cd /workspace/F5-TTS

SRC=ckpts/ghana_pilot/model_${STEP}.safetensors
if [ ! -f "$SRC" ]; then
    echo "checkpoint not found: $SRC -- available:"
    ls ckpts/ghana_pilot/ 2>/dev/null || echo "(none)"
    exit 1
fi

# register the finetune as its own inference model
mkdir -p ckpts/F5TTS_Ghana
cp -f "$SRC" ckpts/F5TTS_Ghana/model_${STEP}.safetensors
cp -f data/ghana_pilot/vocab.txt ckpts/F5TTS_Ghana/vocab.txt

# reference voice: given wav or the first pilot clip
REF_WAV=${2:-$(ls /workspace/ghana_pilot/wav/*.wav | head -1)}
REF_TEXT=$(grep -m1 "$(basename "$REF_WAV" .wav)" /workspace/ghana_pilot/pilot_f5.txt | cut -d'|' -f2)
echo "ref audio: $REF_WAV"
echo "ref text : ${REF_TEXT:0:80}..."

OUT=/workspace/gen/step_${STEP}
mkdir -p "$OUT"
i=0
# skip the 2 header lines, strip the "N. " numbering
tail -n +3 /workspace/ghana_pilot/test_sentences.txt | sed 's/^[0-9]*\. //' | while IFS= read -r sentence; do
    [ -z "$sentence" ] && continue
    i=$((i + 1))
    echo "[$i] $sentence"
    f5-tts_infer-cli \
        --model F5TTS_Ghana \
        --ref_audio "$REF_WAV" \
        --ref_text "$REF_TEXT" \
        --gen_text "$sentence" \
        --output_dir "$OUT" || echo "  [!] generation failed for sentence $i"
done

echo
echo "done -- results in $OUT  (download the folder to listen)"
