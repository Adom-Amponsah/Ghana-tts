# Dataset Overview — ghanaopendata/ghana-english-tts-filtered

_Compiled from Hugging Face Hub API + parquet footer statistics. No full-corpus download was performed._

## Identity

| Field | Value |
|---|---|
| Repo | `ghanaopendata/ghana-english-tts-filtered` |
| Access | Public, no token required |
| Total clips | **303,204** (verified via parquet footers) |
| Total hours | ~1,142 (as reported on the HF page) |
| Audio | 16 kHz, mono, PCM16 WAV embedded in parquet |
| Repo size | ~131 GB across 88 parquet shards (~1.57 GB each) |
| Columns | `audio`, `corrected_text`, `duration_ss`, `mean_speech_prob`, `dbfs` |

## Known metadata defect (why the HF viewer fails)

The shards are named `filtered-train-XXXXX-of-00082.parquet`. The `datasets`
library derives the split name `filtered-train` from the filenames, but split
names must match `^\w+(\.\w+)*$` — hyphens are illegal. Consequences:

- The HF dataset viewer cannot render the dataset.
- `get_dataset_config_names` / `load_dataset(id, split=...)` raise `ValueError`.
- **Workaround**: stream parquet directly via `hf://` URLs or `pyarrow`
  (see `scripts/dataset_info.py`). No "split" exists to select.

Secondary oddities:

- Files `00083`–`00087` are named "of-00082" — appended after initial sharding.
- Shard `00087` has only 881 rows vs ~3,200–4,000 elsewhere.
- Each shard is a **single row group**, so row-level remote reads are
  impossible; any clip extraction costs a full shard download.

## Filter fingerprints (from footer min/max of all 88 shards)

The filtering pipeline left hard floors/ceilings visible in every shard:

| Metric | Floor | Ceiling |
|---|---|---|
| `duration_ss` | **2.28 s** (shard mins range 2.28–4.96) | **15.00 s exactly** |
| `mean_speech_prob` | **0.850** (uniform across all shards) | 0.999 |
| `dbfs` | **−30.0** | −12.0 |

Audit implications:

1. **15.000 s ceiling in every shard** → longer utterances were truncated.
   Clips at exactly ~15 s are strong candidates for mid-sentence cuts.
2. **0.85 speech-probability floor** means "borderline quality" in this
   corpus = clips sitting on that floor; nothing noisier survived.
3. `corrected_text` is still described by the authors as the (corrected) ASR
   transcript, not human transcription — transcript errors are expected.

## Shard statistics uniformity

Footer statistics are near-identical across all 88 shards (same floors, same
15 s ceiling, prob max 0.992–0.999 everywhere). The corpus appears to have
been shuffled before sharding; shards look statistically interchangeable.
This mitigates — but does not prove — representativeness of a small-shard
sample. A second random-shard pass is recommended before trusting
corpus-wide conclusions (see `sample_quality_report.md`).

## Files produced

- `data/manifests/shard_metadata.csv` — per-shard row counts + min/max stats
- `data/manifests/sample_200.csv` — the stratified inspection sample
- `data/samples/*.wav` — 200 clips, 16 kHz mono
- `analysis/clip_qc.csv` — automated DSP QC features + flags
