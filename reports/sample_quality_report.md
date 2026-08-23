# Sample Quality Report — 200-clip stratified inspection sample

_Status: **both audits complete** (pass 1: 200/200, pass 2: 59/60).
Verdict: **GO — ≥ 12 s duration floor confirmed.**_

## 1. Sample design

- **200 clips = 5 strata × 40**: `short` (2–6 s), `medium` (6–10 s),
  `long` (10–15 s), `hi_quality` (speech prob ≥ 0.95), `borderline`
  (lower tail of `mean_speech_prob` / `dbfs` — what the filter barely allowed).
- Drawn from **7 shards**: `00000`, `00001`, `00002`, `00003`, `00024`,
  `00045`, `00066`.
- Selection within shards was seeded (SEED=42) with no row reuse across strata.

### How the 7 shards were picked (honest audit)

`pick_shards()` = **3 lower-tail shards** (lowest `mean_speech_prob_min` /
`dbfs_min`) + **5 shards at quantile positions 0, ¼, ½, ¾, 1** of the
remaining shard list. Two caveats:

1. Because the filter imposed a uniform 0.85 prob floor, **all shards tied**
   on the tail criteria; the tie broke to the first files → `00000–00002`.
   So the "worst shards" intent degenerated to "first three files".
2. The spread picks (`00003, 00024, 00045, 00066, 00087`) are a deliberate
   even sweep across shard indices — but shard index is *not proven* to be
   uncorrelated with source content. If shards were grouped by station or
   recording session, this sample is biased.

**Mitigating evidence:** footer statistics are near-identical across all 88
shards, consistent with pre-sharding shuffling. **Recommended check:** a
second pass with genuinely random shard selection (cheap to add) before
trusting corpus-wide conclusions.

## 2. Sample statistics

| stratum | n | duration range | μ prob | μ dBFS |
|---|---|---|---|---|
| short | 40 | 3.0–5.9 s | 0.906 | −19.2 |
| medium | 40 | 6.1–10.0 s | 0.913 | −21.0 |
| long | 40 | 12.2–15.0 s | 0.908 | −20.8 |
| hi_quality | 40 | 3.3–15.0 s | 0.966 | −21.7 |
| borderline | 40 | 4.6–15.0 s | 0.875 | −23.5 |

Full sample: 200 clips, 16 kHz mono, durations 3.0–14.96 s, median 12.8 s.
**114/200 clips sit in the 12–15 s band** — the corpus is dominated by
near-ceiling-length segments (see the 15.000 s hard ceiling in
`dataset_overview.md`).

## 3. Automated audio QC (`analysis/clip_qc.csv`)

DSP heuristics over all 200 clips; flags are suspects, not verdicts.

| Flag | Count | Meaning |
|---|---|---|
| `abrupt_end` | 35 | last 150 ms hotter than clip average → likely cut mid-utterance |
| `abrupt_start` | 32 | first 150 ms disproportionately hot → likely starts mid-word |
| `possible_music` | 3 | high spectral flatness + low temporal modulation |
| `clipping` | 3 | samples at digital full-scale |
| **any flag** | **68/200 (34 %)** | |

Interpretation so far:

- **~1/3 of clips show cut boundaries.** This is consistent with the 15 s
  truncation ceiling and broadcast segmentation; confirm by ear.
- **No clipping epidemic, no silence-heavy clips** — the loudness/speech-prob
  filters are doing their job.
- 3 possible-music suspects need listening (jingles/beds under speech).

## 4. Human listening session — how to run it

**Marking system (deliberately simple):** verdict `KEEP` / `BORDERLINE` /
`REJECT` + one reason code: `cut_start`, `cut_end`, `multiple_speakers`,
`noise`, `music`, `clipping`, `transcript_mismatch`, `non_ghanaian`,
`unnatural_delivery`, `other`. Reject priority: multiple speakers → cut
start/end → transcript mismatch → heavy noise/music → usable Ghanaian
English → everything else. Don't reject for lack of studio pristine-ness;
pay special attention to how naturally clips start and end (15 s ceiling).

Two equivalent ways to mark:

1. **Excel — `reports/listening_audit.xlsx`**
   - `INSTRUCTIONS`: the 9 listening checks + reason codes.
   - `AUDIT`: all 200 clips (68 auto-flagged first), wav hyperlinks,
     dropdown VERDICT / REASON, colour-coded verdicts — fill while listening.
   - `GOOD` / `BORDERLINE` / `BAD`: live views of the marked rows
     (FILTER formulas; Excel 365/2021 — otherwise filter `AUDIT` by VERDICT).
2. **Browser — `reports/listening_sheet.html`**
   - Same order (68 flagged first), inline player, collapsible 9-check
     reference, verdict + reason per clip, auto-play next.
   - Marks persist in localStorage; **Export marks (CSV)** →
     `listening_marks.csv`. Drop it into `reports/` and I'll compile the
     final verdict statistics and `rejected_samples.csv`.

`reports/listening_sheet.csv` is the same session as a flat spreadsheet.

## 5. Final results (200/200 judged)

### Verdicts

| Verdict | n | Share |
|---|---|---|
| KEEP | 134 | 67.0 % |
| BORDERLINE | 12 | 6.0 % |
| REJECT | 54 | 27.0 % |

Corpus projection (1,142 h): **~765 h keep / ~69 h borderline / ~308 h reject.**

### By stratum

| stratum | KEEP | BORDERLINE | REJECT |
|---|---|---|---|
| short (2–6 s) | 6 | 4 | **30** |
| medium (6–10 s) | 18 | 6 | 16 |
| long (10–15 s) | **39** | 0 | 1 |
| hi_quality | **38** | 0 | 2 |
| borderline | 33 | 2 | 5 |

### Keep rate vs duration — the dominant pattern

| duration | n | KEEP | REJECT | keep rate |
|---|---|---|---|---|
| < 4 s | 17 | 2 | 14 | 12 % |
| 4–6 s | 25 | 5 | 17 | 20 % |
| 6–8 s | 13 | 5 | 6 | 38 % |
| 8–10 s | 25 | 12 | 9 | 48 % |
| 10–12 s | 4 | 3 | 1 | 75 % |
| ≥ 12 s | 116 | 107 | 7 | **92 %** |

Short clips are mostly fragments — cut-off phrases, interjections, jingle
tails, overlap scraps. Long clips are mostly complete utterances. The
quality problem is concentrated almost entirely below ~8 s; **a duration
floor of ~8–10 s lifts the sample keep rate from 67 % to ~85–92 %.**

### Automated DSP QC calibration

The DSP flags turned out to be poor predictors of human judgment:
flagged clips were kept at 63 % (43/68) vs 69 % for unflagged (91/132),
and 28 % of *unflagged* clips were rejected. The energy-based
start/end heuristics over-flag broadcast speech; the real failure modes
(fragments, multi-speaker, transcript issues) live in the short clips and
are not energy-visible. **For full-corpus cleaning, use duration + length
filters first, not these DSP flags.**

### Caveats

- REASON column was not filled during the audit; the duration analysis
  above substitutes for most of that signal, but the exact reject-cause
  mix (multi-speaker vs transcript vs music) is not recorded.
- The 200-clip sample came from 7 shards (see section 1); a second
  random-shard pass was not run. Shard-footer uniformity makes a large
  surprise unlikely.
- Sample durations are stratified, so the duration table above is **not**
  the corpus duration mix. To project filtered-corpus hours we need the
  true corpus duration distribution (cheap: metadata-only read of the
  `duration_ss` column across all 88 shards).

## 6. Verdict & next steps

**GO.** ~765 h usable as-is; potentially fewer-but-cleaner hours with an
~8–10 s duration floor. The dataset is training-grade material for a
Ghanaian-English TTS voice.

1. Profile corpus durations (metadata-only, all shards) → hours remaining
   after an 8 s / 10 s / 12 s floor.
2. Decide filter policy, then full 131 GB download (resumable `hf download`).
3. Extract + build training manifest from kept rows.

### Files produced

- `reports/kept_samples.csv` — 134 KEEP clips with metadata
- `reports/rejected_samples.csv` — 54 REJECT clips with metadata
- `reports/borderline_samples.csv` — 12 BORDERLINE clips
- `reports/listening_audit.xlsx` — the filled audit workbook (source of truth)

## 7. Full-corpus duration profile (all 88 shards, metadata-only)

Verified totals: **303,204 clips, 1,142.3 h** (matches the HF page exactly).
Min 2.28 s · median **13.76 s** · max 15.00 s. The corpus is overwhelmingly
long clips — short fragments are a tiny sliver:

| floor | clips | clips % | hours | hours % |
|---|---|---|---|---|
| none | 303,204 | 100.0 | 1,142.3 | 100.0 |
| ≥ 8 s | 301,185 | 99.3 | 1,139.3 | 99.7 |
| ≥ 10 s | 299,826 | 98.9 | 1,135.9 | 99.4 |
| **≥ 12 s** | **284,653** | **93.9** | **1,087.7** | **95.2** |
| ≥ 13 s | 230,611 | 76.1 | 898.8 | 78.7 |
| ≥ 14 s | 123,152 | 40.6 | 495.7 | 43.4 |

### Recommended filter: **≥ 12 s** — CONFIRMED by pass 2

Combining the audit keep rates with the corpus duration mix:

- **≥ 12 s**: 1,087.7 h × 92 % keep ≈ **~1,000 h clean** — costs only 4.8 %
  of corpus hours.
- ≥ 10 s: adds ~48 h at ~75 % keep → ~1,035 h clean; marginal gain, more risk.
- no floor: ~765 h expected clean plus ~300 h of mostly-short junk inside.

The ≥ 12 s floor is nearly free and lands on the 92 %-keep region of the
audit curve. Estimated final training set: **≈ 1,000 hours of clean
Ghanaian-English speech (~284 k clips).**

- `data/manifests/all_durations.csv` — every clip's duration (303,204 rows)
- `data/manifests/duration_profile.csv` — the survival table above

**Remaining steps:** confirm filter → full 131 GB download (resumable
`hf download`) → extract + build training manifest of rows with
`duration_ss ≥ 12`.

## 8. Pass 2 — random-shard truncation check (59/60 judged)

Design: 8 uniform-random shards never touched by pass 1 (00034, 00039,
00049, 00053, 00063, 00075, 00080, 00086); 48 uniform-random clips from
the 12–15 s band + 12 stratified ceiling-targeted clips (14.8–14.96 s).

| block | KEEP | BORDERLINE | REJECT |
|---|---|---|---|
| ceiling_target (n=12) | **9 (75 %)** | 1 | 2 |
| uniform_random (n=47) | **38 (81 %)** | 6 | 3 (6 %) |

Readings:

1. **Truncation risk is real but bounded** — 75 % of clips pressed against
   the 15 s cap still end cleanly. The segmenter mostly found sentence
   boundaries before the cap (only 0.9 % of the corpus sits at exactly
   15.000 s).
2. **Independent keep rate at ≥ 12 s ≈ 81 % keep + 13 % borderline.**
   Below pass 1's 92 %, but with n=47 the 95 % CI is ~68–90 % — the two
   passes are statistically compatible. Pooled best estimate: **~85–88 %
   keep, ~10 % borderline, ~5–8 % reject.**
3. Representativeness gap closed: random shards behave like pass-1 shards.

### Final projection (≥ 12 s floor, 1,087.7 h)

- keep: **~925–955 h** (conservative 85 % → 925 h)
- borderline: ~110 h (usable with tolerance)
- reject: ~55–90 h

**The floor decision stands: ≥ 12 s. The dataset is confirmed
training-grade. Ready for the full download whenever you are.**

Files: `data/manifests/sample_pass2.csv` (60 clips),
`reports/pass2_truncation_audit.xlsx` (filled workbook),
`data/samples_pass2/*.wav`.

## 9. Pilot phase — architecture bake-off (in progress)

Decision: do **not** download the full 131 GB yet. Build a ~15 h pilot
from >= 12 s clips and compare two architectures before committing
storage and compute.

| candidate | stack | effort |
|---|---|---|
| A. YarnGPT-style | SmolLM2-360M + audio tokens (WavTokenizer), OuteTTS-style training | research build |
| B. F5-TTS fine-tune | established fine-tune workflow, `audio\|text\|duration` manifest | quick baseline |

Pilot dataset (built by `scripts/build_pilot_dataset.py`): ~15 h,
>= 12 s only, ~495 uniform-random clips from each of 8 fresh shards
(seed 2026; all 15 audit shards excluded). Manifests in both formats:
`data/manifests/pilot.csv` and `data/manifests/pilot_f5.txt`.
Eval set: `data/test_sentences.txt` (15 sentences covering Ghanaian
place names, loanwords, currency, numbers, questions).

Scale-up ladder after a winner is chosen: 100 h -> 300 h -> ~1,000 h.
