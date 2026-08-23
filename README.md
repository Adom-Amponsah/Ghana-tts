# Ghana-tts

Quality audit of the Hugging Face dataset
[`ghanaopendata/ghana-english-tts-filtered`](https://huggingface.co/datasets/ghanaopendata/ghana-english-tts-filtered)
(~303,204 clips / ~1,142 h of Ghanaian news-broadcast speech, 16 kHz mono)
to decide whether it is training-grade material for a Ghanaian-English TTS voice.

**Verdict: GO.** With a `duration_ss >= 12 s` filter the corpus yields an
estimated **~925-955 h of clean speech** (~85-88% keep rate, verified by two
independent human listening audits).

## What's here

| Path | Content |
|---|---|
| `scripts/` | every step, reproducible: dataset discovery, stratified sampling, DSP QC, listening-sheet builders, duration profiling, verdict compilers |
| `reports/` | `dataset_overview.md`, `sample_quality_report.md` (full audit results), audit workbooks, listening sheets |
| `data/manifests/` | per-shard footer stats, sample manifests, full-corpus duration profile (`.csv.gz`) |
| `analysis/` | automated DSP QC features for the 200-clip sample |

Audio samples are **not** included (gitignored); they are re-extractable from
the HF dataset via `scripts/build_sample.py` / `scripts/random_pass_12s.py`.

## Key findings

- The HF viewer fails on this dataset: shard names produce the illegal split
  name `filtered-train`. All access goes through direct parquet reads
  (`hf://` URLs) - see `scripts/dataset_info.py`.
- Filter fingerprints: hard 15.000 s ceiling, 0.85 speech-prob floor, -30 dBFS
  floor. Only 0.9% of clips sit exactly at the ceiling.
- Clip duration is the dominant quality predictor: 12% keep below 4 s vs
  92% keep at >= 12 s (pass 1, n=200). Pass 2 on random shards confirmed
  81% keep + 13% borderline in the 12-15 s band.
- DSP energy heuristics are poor predictors of human judgment; use the
  duration floor instead.

## Environment

Python 3.13 venv (`.venv/`, gitignored). Deps: `datasets`, `huggingface_hub`,
`pandas`, `pyarrow`, `soundfile`, `librosa`, `numpy`, `matplotlib`, `tqdm`,
`openpyxl`.
