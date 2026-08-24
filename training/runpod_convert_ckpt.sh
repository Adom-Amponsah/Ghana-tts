#!/bin/bash
# ============================================================================
# Convert a F5-TTS .pt training checkpoint to .safetensors for inference.
#
# Usage:  bash runpod_convert_ckpt.sh <step>
#   e.g.  bash runpod_convert_ckpt.sh 27500
#
# Reads:   ckpts/ghana_pilot/model_<step>.pt
# Writes:  ckpts/ghana_pilot/model_<step>.safetensors
# ============================================================================
set -e
STEP=${1:?usage: runpod_convert_ckpt.sh <step>}
cd /workspace/F5-TTS

SRC=ckpts/ghana_pilot/model_${STEP}.pt
if [ ! -f "$SRC" ]; then
    echo "checkpoint not found: $SRC -- available:"
    ls ckpts/ghana_pilot/*.pt 2>/dev/null || echo "(none)"
    exit 1
fi

DST=ckpts/ghana_pilot/model_${STEP}.safetensors

python -c "
import torch
from safetensors.torch import save_file
from safetensors import safe_open

ckpt = torch.load('${SRC}', map_location='cpu', weights_only=False)
print('checkpoint top-level keys:', list(ckpt.keys()))

# F5-TTS saves ema_model_state_dict — that's what the pretrained .safetensors uses.
# Filter out non-weight keys like 'initted' and 'step'.
ema = ckpt['ema_model_state_dict']
state = {k: v.contiguous() for k, v in ema.items() if k.startswith('ema_model.')}
print(f'Extracted {len(state)} ema_model.* tensors (filtered out metadata keys)')

# Compare against pretrained model keys
with safe_open('ckpts/ghana_pilot/pretrained_model_1250000.safetensors', framework='pt') as f:
    pretrained_keys = set(f.keys())

our_keys = set(state.keys())
missing = pretrained_keys - our_keys      # in pretrained, not in ours
extra   = our_keys - pretrained_keys      # in ours, not in pretrained

if missing:
    print(f'WARNING: {len(missing)} keys in pretrained but NOT in checkpoint:')
    for k in sorted(missing):
        print(f'  - {k}')
if extra:
    print(f'WARNING: {len(extra)} keys in checkpoint but NOT in pretrained:')
    for k in sorted(extra):
        print(f'  - {k}')
if not missing and not extra:
    print('Key match: PERFECT — checkpoint keys match pretrained exactly')

# Also verify tensor shapes match for shared keys
with safe_open('ckpts/ghana_pilot/pretrained_model_1250000.safetensors', framework='pt') as f:
    shape_mismatches = []
    for k in pretrained_keys & our_keys:
        pt_shape = f.get_tensor(k).shape
        our_shape = state[k].shape
        if pt_shape != our_shape:
            shape_mismatches.append((k, pt_shape, our_shape))
    if shape_mismatches:
        print(f'WARNING: {len(shape_mismatches)} shape mismatches:')
        for k, ps, os in shape_mismatches[:10]:
            print(f'  {k}: pretrained {ps} vs checkpoint {os}')
    else:
        print('Shape match: PERFECT — all shared keys have identical shapes')

save_file(state, '${DST}')
print(f'Saved {len(state)} tensors to ${DST}')
"
