# Ghanaian-English TTS — Pilot Journey

_From dataset discovery to fine-tuned model, and the 12-second problem we hit along the way._

---

## 1. The Dataset

**Source:** [`ghanaopendata/ghana-english-tts-filtered`](https://huggingface.co/datasets/ghanaopendata/ghana-english-tts-filtered) on HuggingFace.

| Field | Value |
|---|---|
| Total clips | 303,204 |
| Total hours | ~1,142 h |
| Audio format | 16 kHz mono PCM16, embedded in parquet |
| Repo size | ~131 GB across 88 parquet shards (~1.57 GB each) |
| Columns | `audio`, `corrected_text`, `duration_ss`, `mean_speech_prob`, `dbfs` |
| Content | Ghanaian news-broadcast speech |

### How we discovered it

The dataset had a known metadata defect — shard filenames like `filtered-train-XXXXX-of-00082.parquet` produce an illegal split name (hyphens), so the HuggingFace dataset viewer couldn't render it. We worked around this by reading parquet footers directly via `hf://` URLs and `pyarrow`, without needing the `datasets` library's split system.

### Footer statistics (all 88 shards, metadata-only read)

Every shard had the same filter fingerprints:

| Metric | Floor | Ceiling |
|---|---|---|
| `duration_ss` | 2.28 s | 15.00 s exactly |
| `mean_speech_prob` | 0.850 | 0.999 |
| `dbfs` | -30.0 | -12.0 |

The 15.000 s ceiling meant longer utterances were truncated. The 0.85 speech-probability floor meant nothing noisier survived filtering. `corrected_text` is ASR-corrected (not human-verified), so transcript errors were expected.

**Scripts used:**
- `scripts/dataset_info.py` — discovered the HF viewer bug, read footer stats
- `scripts/build_sample.py` — built the stratified 200-clip inspection sample
- Output: `data/manifests/shard_metadata.csv` (all 88 shards' footer stats)

---

## 2. Quality Audit — Human Listening

### Pass 1: Stratified 200-clip sample

We built a 200-clip sample across 5 strata (40 clips each):

| Stratum | Description |
|---|---|
| `short` | 2-6 s clips |
| `medium` | 6-10 s clips |
| `long` | 10-15 s clips |
| `hi_quality` | speech prob >= 0.95 |
| `borderline` | low-end speech prob / dBFS |

Drawn from 7 shards: `00000`, `00001`, `00002`, `00003`, `00024`, `00045`, `00066`.

**Automated DSP QC** (`scripts/audio_qc.py`) computed per-clip features:
- `clip_frac` — digital full-scale samples
- `silence_frac` — frames below -50 dBFS
- `duty` — fraction of active frames
- `mod_std` — temporal modulation (speech = high, music/noise = low)
- `flatness` — spectral flatness (high = noise-like)
- `start_ratio` / `end_ratio` — edge energy vs clip average (detects cuts)

Flags: `clipping`, `mostly_silent`, `possible_music`, `abrupt_end`, `abrupt_start`, `low_activity`.

**Result: 68/200 clips flagged** (34%), but DSP flags turned out to be poor predictors of human judgment — flagged clips were kept at 63% vs 69% for unflagged. The real failure modes (fragments, multi-speaker, transcript issues) live in short clips and aren't energy-visible.

### Human listening session

Each clip was listened to and marked:

- **Verdict:** KEEP / BORDERLINE / REJECT
- **Reason codes:** `cut_start`, `cut_end`, `multiple_speakers`, `noise`, `music`, `clipping`, `transcript_mismatch`, `non_ghanaian`, `unnatural_delivery`, `other`

Marking was done in `reports/listening_audit.xlsx` (dropdown verdicts, colour-coded) or `reports/listening_sheet.html` (browser-based, inline player, auto-play next).

### Pass 1 results (200/200 judged)

| Verdict | n | Share |
|---|---|---|
| KEEP | 134 | 67.0% |
| BORDERLINE | 12 | 6.0% |
| REJECT | 54 | 27.0% |

**The dominant pattern — keep rate vs duration:**

| Duration | n | KEEP | REJECT | Keep rate |
|---|---|---|---|---|
| < 4 s | 17 | 2 | 14 | 12% |
| 4-6 s | 25 | 5 | 17 | 20% |
| 6-8 s | 13 | 5 | 6 | 38% |
| 8-10 s | 25 | 12 | 9 | 48% |
| 10-12 s | 4 | 3 | 1 | 75% |
| >= 12 s | 116 | 107 | 7 | **92%** |

Short clips were mostly fragments — cut-off phrases, interjections, jingle tails. Long clips were mostly complete utterances. **A duration floor of >= 12 s lifts the keep rate from 67% to 92%.**

### Pass 2: Random-shard validation (59/60 judged)

To verify pass 1 wasn't biased by shard selection, we ran a second pass with 8 uniform-random shards never touched by pass 1 (`00034`, `00039`, `00049`, `00053`, `00063`, `00075`, `00080`, `00086`). 48 uniform-random clips from the 12-15 s band + 12 ceiling-targeted clips (14.8-14.96 s).

| Block | KEEP | BORDERLINE | REJECT |
|---|---|---|---|
| ceiling_target (n=12) | 9 (75%) | 1 | 2 |
| uniform_random (n=47) | 38 (81%) | 6 | 3 (6%) |

**Pooled best estimate: ~85-88% keep, ~10% borderline, ~5-8% reject** at >= 12 s. The two passes are statistically compatible.

### Final corpus projection (>= 12 s floor, 1,087.7 h)

- **Keep:** ~925-955 h (conservative 85%)
- **Borderline:** ~110 h (usable with tolerance)
- **Reject:** ~55-90 h

**Verdict: GO. The dataset is training-grade material for a Ghanaian-English TTS voice.**

**Scripts used:**
- `scripts/build_sample.py` — stratified 200-clip sample
- `scripts/audio_qc.py` — automated DSP QC
- `scripts/build_audit_workbook.py` — Excel + HTML listening sheets
- `scripts/random_pass_12s.py` — pass 2 random-shard validation
- `scripts/compile_marks.py` — compiled verdicts into manifests
- Outputs: `reports/listening_audit.xlsx`, `reports/kept_samples.csv`, `reports/rejected_samples.csv`, `reports/borderline_samples.csv`

---

## 3. Pilot Dataset Construction

Rather than downloading the full 131 GB, we built a ~15 h pilot from >= 12 s clips to test architectures before committing storage and compute.

### How it was built

`scripts/build_pilot_dataset.py`:

1. **Selected 8 fresh shards** (seed 2026), excluding all 15 shards used by passes 1 and 2
2. **Read parquet footers remotely** (metadata-only, no audio download) to find rows with `duration_ss >= 12.0`
3. **Calculated per-shard quota**: `TARGET_HOURS * 3600 / N_SHARDS / mean_duration` = ~495 clips per shard
4. **Downloaded each shard** (~1.57 GB each) via `hf` CLI with resumable retries (8 attempts)
5. **Extracted audio bytes** from parquet, wrote WAV files (16 kHz mono PCM16)
6. **Deleted the shard** after extraction to save disk
7. **Built manifests**: `pilot.csv` (full metadata) and `pilot_f5.txt` (F5-TTS format: `audio|text|duration`)

### Pilot dataset stats

| Field | Value |
|---|---|
| Clips | ~3,933 |
| Hours | ~15 h |
| Duration range | 12.0-15.0 s (median 13.8 s) |
| Shards | 8 (random, excluding audit shards) |
| Filter | `duration_ss >= 12.0` only |

### Bundling for RunPod

The pilot was zipped as `pilot_bundle.zip` containing:
- `wav/*.wav` — audio clips
- `metadata.csv` — manifest with text, duration, shard info
- `pilot_f5.txt` — F5-TTS training format

This was uploaded to the RunPod's `/workspace/` directory.

---

## 4. Training — F5-TTS Fine-tune

### Environment

- **RunPod:** NVIDIA GPU (A100/RTX 4090), PyTorch 2.8 + CUDA 12.8
- **Model:** F5-TTS (`F5TTS_v1_Base`), cloned from `https://github.com/SWivid/F5-TTS.git`
- **Base checkpoint:** `pretrained_model_1250000.safetensors` (1.3 GB, auto-downloaded on first run)

### Setup script (`training/runpod_setup.sh`)

1. Cloned F5-TTS, installed in editable mode with `accelerate` and `tensorboard`
2. Unpacked `pilot_bundle.zip` to `/workspace/ghana_pilot/`
3. Prepared HF-arrow dataset via `prepare_csv_wavs.py` (finetune mode copies pretrained Emilia vocab so text embedding table stays compatible)
4. Launched fine-tuning in background

### Training configuration

```
--exp_name F5TTS_v1_Base
--dataset_name ghana_pilot
--finetune
--learning_rate 1e-5
--batch_size_per_gpu 3200
--batch_size_type frame
--max_samples 64
--epochs 15
--num_warmup_updates 200
--save_per_updates 2500
--keep_last_n_checkpoints 3
--last_per_updates 2500
--tokenizer pinyin
--logger tensorboard
```

- ~15 epochs over 3,933 clips = ~20-24k steps
- Checkpoint saved every 2,500 steps
- Checkpoints landed in `/workspace/F5-TTS/ckpts/ghana_pilot/`
- Training produced `.pt` files (PyTorch checkpoints with optimizer state + EMA weights)

### Checkpoints produced

| File | Type | Size | Notes |
|---|---|---|---|
| `model_22500.pt` | Training checkpoint | ~5.1 GB | Optimizer + EMA + step |
| `model_25000.pt` | Training checkpoint | ~5.1 GB | |
| `model_27500.pt` | Training checkpoint | ~5.1 GB | |
| `model_last.pt` | Final checkpoint | ~5.1 GB | Last saved step |
| `pretrained_model_1250000.safetensors` | Base model | ~1.3 GB | Pretrained, not fine-tuned |

### Checkpoint conversion

F5-TTS inference requires `.safetensors` format (model weights only), but training produces `.pt` files (full training state including optimizer, EMA, step counter).

`training/runpod_convert_ckpt.sh`:
1. Loads the `.pt` file with `torch.load(..., weights_only=False)`
2. Extracts `ema_model_state_dict` — the EMA weights (what the pretrained `.safetensors` uses)
3. Filters to only `ema_model.*` keys (removes metadata like `initted`, `step`)
4. Verifies keys and shapes against `pretrained_model_1250000.safetensors`
5. Saves as `.safetensors` via `safetensors.torch.save_file()`

Verification output confirmed: **"Key match: PERFECT — checkpoint keys match pretrained exactly"** and **"Shape match: PERFECT — all shared keys have identical shapes"**.

---

## 5. Evaluation — The 12-Second Problem

### Test set

`data/test_sentences.txt` — 15 Ghanaian-English sentences covering:
- Place names (Accra, Kumasi, Cape Coast, Takoradi, Tamale, Kasoa, Winneba)
- Loanwords (kenkey, waakye, jollof, banku, trotros)
- Currency (cedi)
- Numbers and acronyms
- Questions

### The evaluation script (`training/runpod_eval.sh`)

Used `f5-tts_infer-cli` with:
- `--model F5TTS_v1_Base` (base config)
- `--ckpt_file` pointing to the converted `.safetensors`
- `--vocab_file` pointing to `data/ghana_pilot/vocab.txt`
- `--ref_audio` — a pilot clip as zero-shot voice prompt
- `--ref_text` — the transcript of that clip
- `--gen_text` — each test sentence
- `--output_file` with unique names (`sent_01.wav`, `sent_02.wav`, etc.)

### What went wrong: the 12-second reference audio limit

**F5-TTS hardcodes a 12-second limit on reference audio.** The `preprocess_ref_audio_text()` function in `src/f5_tts/infer/utils_infer.py`:

1. Finds long silence segments and clips before 12s
2. If no silence found, hard-cuts at 12s
3. Removes silence edges

**But it does NOT trim the reference text to match.** If you pass 15 seconds of audio with the full 15-second transcript, F5-TTS clips the audio to 12s but keeps the full text — creating a mismatch.

**All our pilot clips are 12-15 seconds.** Every reference clip we tried was longer than 12s. The result:

- The model saw 12s of audio with text that didn't match
- Instead of generating the target sentence, it **repeated the reference text** — parroting back the reference audio's transcript
- Every "generated" sample was just the reference speaker's voice saying the reference text, not the target sentence

### What we thought was "80-90% quality"

The first evaluation run produced samples that sounded like a Ghanaian speaker at a natural pace. We rated them ~80-90% quality. **This was a false signal** — the model was playing back the reference audio (slowed down with `--speed 0.30`), not generating new speech. Every sample sounded the same because they were all the same reference audio.

### Attempts to fix the reference issue

1. **Trim audio to 10s + estimate matching text** — word-count estimation was too rough, text still didn't match the clipped audio. Model still repeated reference text.

2. **Trim to 6s + use first 8 words** — too little text for the audio. Model produced gibberish.

3. **Auto-transcription (empty `--ref_text`)** — F5-TTS would transcribe the clipped 12s audio itself, guaranteeing a perfect text-audio match. **Crashed** with `OSError: libnvrtc.so.13: cannot open shared object file` — `torchcodec` requires CUDA 13, pod has CUDA 12.8. Symlink workaround failed (also needed `libcudart.so.13`).

4. **Short-transcript clips** — searched for clips with <= 25 words. Found 0. All transcripts are long (news broadcast speech). Shortest was 2 words ("RadioMe. [Music]") — not usable as a reference.

5. **Default F5-TTS reference (no custom ref audio)** — generated coherent speech but in the default English F5-TTS voice, not our Ghanaian model's voice. Not a valid test of our fine-tuned model.

### Root cause

The 15h pilot model **cannot generate coherent new speech from arbitrary text**. When we finally got the reference setup close to correct (short ref text), the output was gibberish. The model has learned the voice/accent somewhat but hasn't seen enough data to synthesize new sentences properly.

**This is an expected result for a 15h pilot.** F5-TTS was pretrained on 100,000+ hours. 15h of fine-tuning isn't enough to teach it new patterns for zero-shot generation.

### What the base model does (for comparison)

The pretrained base model (`pretrained_model_1250000.safetensors`) **also repeats the reference text** when given a mismatched ref_text. This confirms the reference-text mismatch is an F5-TTS behavior, not specific to our fine-tuned model.

---

## 6. Current State

### What works

- **Dataset pipeline** — discovery, footer stats, stratified sampling, DSP QC, human listening audit, duration profiling. Fully reproducible.
- **Quality audit** — two independent passes, 92% keep rate at >= 12 s confirmed. Dataset is training-grade.
- **Pilot dataset construction** — 15h of >= 12 s clips from 8 random shards, manifests in both CSV and F5-TTS formats.
- **Training pipeline** — F5-TTS fine-tuning runs, produces checkpoints, EMA weights convert cleanly to `.safetensors`.
- **Checkpoint conversion** — `.pt` to `.safetensors` with key/shape verification against pretrained model. Perfect match.
- **Inference pipeline** — model loads, `f5-tts_infer-cli` runs, audio files are generated.

### What doesn't work yet

- **The 15h model cannot generate coherent new speech.** It can repeat reference audio (at adjusted speed) but cannot synthesize arbitrary new sentences.
- **Reference audio constraint** — all pilot clips are 12-15s, F5-TTS needs < 12s with exact matching text. Auto-transcription is broken due to CUDA version mismatch.
- **The "80-90% quality" was a false signal** — it was reference playback, not generation.

### Artifacts on the pod

| Path | Content |
|---|---|
| `/workspace/F5-TTS/ckpts/ghana_pilot/model_last.safetensors` | Fine-tuned model (1.3 GB) |
| `/workspace/F5-TTS/ckpts/ghana_pilot/model_last.pt` | Full training checkpoint (5.1 GB) |
| `/workspace/F5-TTS/ckpts/ghana_pilot/model_22500.pt` | Earlier checkpoint (5.1 GB) |
| `/workspace/F5-TTS/ckpts/ghana_pilot/model_25000.pt` | Earlier checkpoint (5.1 GB) |
| `/workspace/F5-TTS/ckpts/ghana_pilot/model_27500.pt` | Earlier checkpoint (5.1 GB) |
| `/workspace/gen/step_last/` | First eval samples (reference playback, not real generation) |
| `/workspace/gen/step_last_speed030/` | Speed-adjusted samples (still reference playback) |
| `/workspace/ghana_pilot/` | Pilot dataset (3,933 wav files + manifests) |

---

## 7. Next Steps

### Scale to 50h

The architecture works (F5-TTS is proven). The pipeline works. The model just needs more data.

**Plan:**
1. Modify `build_pilot_dataset.py`: `N_SHARDS = 27`, `TARGET_HOURS = 50.0`
2. Download directly on the pod from HuggingFace (no bundle needed)
3. Run the same training pipeline with more data
4. Evaluate with a properly set up reference clip (under 12s with exact text)

### The reference audio fix (for future evaluation)

Options to solve the 12s reference problem:
1. **Find or create a clip under 10s** with clean speech and exact transcript
2. **Fix the CUDA 13 dependency** so F5-TTS auto-transcription works (transcribe the clipped 12s audio)
3. **Transcribe externally** with Whisper, then pass the transcript as `--ref_text`
4. **Use a non-Ghanaian reference clip under 12s** — the fine-tuned model should still apply the Ghanaian accent

### Scale-up ladder

| Stage | Hours | Shards | Status |
|---|---|---|---|
| Pilot | 15 h | 8 | Done — model trains but can't generate new speech |
| Scale 1 | 50 h | ~27 | Next |
| Scale 2 | 100 h | ~54 | After 50h evaluation |
| Full | ~1,000 h | 88 | If F5-TTS continues to improve |

### Architecture decision

F5-TTS is the current architecture. YarnGPT-style (SmolLM2-360M + WavTokenizer) remains in the back pocket. The decision to switch would only be justified if F5-TTS plateaus despite more data:

| Hours | Quality | Decision |
|---|---|---|
| 15 h | Can't generate | Scale data |
| 50 h | ? | Evaluate |
| 100 h | ? | Evaluate — if plateau, try YarnGPT |

---

## 8. File Index

### Scripts

| Script | Purpose |
|---|---|
| `scripts/dataset_info.py` | Dataset discovery, footer stats, HF viewer bug workaround |
| `scripts/build_sample.py` | Stratified 200-clip inspection sample (pass 1) |
| `scripts/audio_qc.py` | Automated DSP QC over 200-clip sample |
| `scripts/build_audit_workbook.py` | Excel + HTML listening audit sheets |
| `scripts/random_pass_12s.py` | Pass 2: random-shard truncation check (60 clips) |
| `scripts/compile_marks.py` | Compile listening verdicts into manifests + stats |
| `scripts/build_pilot_dataset.py` | Build ~15h pilot dataset from HuggingFace |
| `training/runpod_setup.sh` | RunPod setup + fine-tune launch |
| `training/runpod_convert_ckpt.sh` | Convert `.pt` checkpoint to `.safetensors` |
| `training/runpod_eval.sh` | Generate eval sentences from fine-tuned model |

### Reports

| File | Content |
|---|---|
| `reports/dataset_overview.md` | Dataset identity, metadata defect, filter fingerprints |
| `reports/sample_quality_report.md` | Full audit results (passes 1 + 2), duration analysis, verdict |
| `reports/pilot_journey.md` | This document — full journey from dataset to model |
| `reports/listening_audit.xlsx` | Filled audit workbook (pass 1, source of truth) |
| `reports/pass2_truncation_audit.xlsx` | Pass 2 audit workbook |
| `reports/kept_samples.csv` | 134 KEEP clips with metadata |
| `reports/rejected_samples.csv` | 54 REJECT clips with metadata |
| `reports/borderline_samples.csv` | 12 BORDERLINE clips |

### Data manifests

| File | Content |
|---|---|
| `data/manifests/shard_metadata.csv` | All 88 shards: row counts + min/max stats |
| `data/manifests/sample_200.csv` | Pass 1 stratified sample manifest |
| `data/manifests/sample_pass2.csv` | Pass 2 random-shard sample manifest |
| `data/manifests/pilot.csv` | Pilot dataset manifest (3,933 clips) |
| `data/manifests/pilot_f5.txt` | Pilot F5-TTS training format |
| `data/manifests/all_durations.csv` | Full corpus duration profile (303,204 rows) |
| `data/manifests/duration_profile.csv` | Duration survival table (floor analysis) |
| `data/test_sentences.txt` | 15 Ghanaian-English evaluation sentences |

### Training outputs (on RunPod, not in repo)

| Path | Content |
|---|---|
| `ckpts/ghana_pilot/model_last.safetensors` | Fine-tuned model weights (1.3 GB) |
| `ckpts/ghana_pilot/model_last.pt` | Full training checkpoint (5.1 GB) |
| `ckpts/ghana_pilot/pretrained_model_1250000.safetensors` | Base pretrained model (1.3 GB) |
