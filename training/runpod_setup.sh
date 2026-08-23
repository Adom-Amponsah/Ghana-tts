#!/bin/bash
# ============================================================================
# RunPod setup + fine-tune launch for the Ghana-English F5-TTS pilot.
# Run from the pod terminal AFTER uploading pilot_bundle.zip to /workspace.
# Re-runnable: cloning/prep are skipped if already done; training resumes.
# ============================================================================
set -e
cd /workspace

# ---- 1. Install F5-TTS (editable; pod already has torch 2.8 cu128) --------
if [ ! -d F5-TTS ]; then
    git clone https://github.com/SWivid/F5-TTS.git
fi
cd F5-TTS
pip install -e . accelerate tensorboard

# ---- 2. Unpack the pilot dataset -------------------------------------------
if [ ! -d /workspace/ghana_pilot/wav ]; then
    mkdir -p /workspace/ghana_pilot
    unzip -q -o /workspace/pilot_bundle.zip -d /workspace/ghana_pilot
fi
echo "wav files: $(ls /workspace/ghana_pilot/wav | wc -l)"

# ---- 3. Prepare the HF-arrow dataset (skips if already prepared) ----------
# finetune mode (default) copies the pretrained Emilia vocab, so the text
# embedding table stays compatible with the base checkpoint.
if [ ! -f data/ghana_pilot/raw.arrow ]; then
    python src/f5_tts/train/datasets/prepare_csv_wavs.py \
        /workspace/ghana_pilot/metadata.csv data/ghana_pilot
fi

# ---- 4. Launch fine-tuning (background, logged) ----------------------------
# ~15 epochs over 3,933 clips ~ 20-24k steps. Checkpoint every 2,500 steps.
# Base checkpoint (model_1250000.safetensors) auto-downloads on first run.
nohup python src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name ghana_pilot \
    --finetune \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 3200 \
    --batch_size_type frame \
    --max_samples 64 \
    --epochs 15 \
    --num_warmup_updates 200 \
    --save_per_updates 2500 \
    --keep_last_n_checkpoints 3 \
    --last_per_updates 2500 \
    --tokenizer pinyin \
    --logger tensorboard \
    > /workspace/train.log 2>&1 &

echo
echo "Training launched in background. Watch with:   tail -f /workspace/train.log"
echo "Checkpoints land in:  /workspace/F5-TTS/ckpts/ghana_pilot/"
echo "If you hit OOM, kill and rerun with  --batch_size_per_gpu 2200"
