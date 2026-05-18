# naturelm-cats

**Extending [NatureLM-audio](https://github.com/earthspecies/NatureLM-audio) (Earth Species Project) to cat vocalization understanding, with a more powerful reasoning LLM bridge.**

A soft fork of `earthspecies/NatureLM-audio` that adds:

1. A **bridge from NatureLM-audio to GPT-OSS-20B** via `llama-server` — getting structured behavioral reasoning on top of NatureLM's bioacoustic captions, running on a single consumer GPU (RTX 4070 Laptop, 8 GB).
2. A **cat-vocalization context classifier** trained on BEATs features from [CatMeows](https://zenodo.org/records/4008297) (Ludovico et al. 2020) — predicting *brushing*, *isolation*, or *waiting-for-food* context from raw audio.

Both pieces work end-to-end against arbitrary cat audio today.

> The original NatureLM-audio README is preserved at [`README_UPSTREAM.md`](README_UPSTREAM.md).

---

## Pipeline

```
audio (any cat sound, wav/mp3/flac/ogg)
  │
  ├─► NatureLM-audio (CPU)
  │     BEATs + Q-Former + Llama-3.1-8B + LoRA
  │     → generic bioacoustic caption: "Domestic cats vocalizing."
  │
  └─► BEATs encoder + statistical pooling (mean+std+max) + trained MLP head (CPU)
        → cat-context softmax:
          {brushing: 86.2%, isolation: 10.2%, waiting_for_food: 3.6%}
  │
  ▼  merge both signals
GPT-OSS-20B Q4_K_M (RTX 4070, llama-server :9002)
  → structured behavioral analysis with confidence calibration
```

### Bridge variants

| Script | Inputs to GPT-OSS-20B | When to use |
|---|---|---|
| `scripts/bridge_caption.py` | NatureLM caption (single generic query) | Simplest. Pre-classifier baseline. |
| `scripts/bridge_with_context.py` | Caption + cat-context classifier probs | Default. 82.2% CV classifier grounding. |
| **`scripts/bridge_multiquery.py`** | **3 NatureLM queries (acoustic / emotion / sentence) + classifier** | **Richest. GPT-OSS-20B can flag caption-classifier disagreements.** |

The multi-query bridge runs NatureLM three times per clip (~3× the audio inference cost, ~15s warm) but gives GPT-OSS-20B enough independent signals to reason about *which* caption to trust when they disagree. Probe results in `scripts/probe_naturelm_queries.py` showed:

- **Acoustic-characteristics** is the most informative query — distinct per-clip outputs (pitch, rhythm, duration, intensity).
- **Emotion** and **if-a-sentence** queries are partially noisy — NatureLM has default-y outputs ("Distressed", "I want to be petted") that surface inconsistently across clips. The multi-query bridge still uses them because *when they engage*, they add a useful independent signal, and GPT-OSS-20B correctly weighs them against the classifier (committing when classifier is confident, flagging the noise when it isn't).

### Sample output (ESC-50 cat clip)

> *Generic caption:* `"Domestic cats vocalizing."`
> *Classifier:* `brushing (86.2%), isolation (10.2%), waiting_for_food (3.6%)`
>
> **GPT-OSS-20B reasoning:** Confident interpretation with explicit uncertainty calibration based on the classifier's softmax distribution. The reasoning is grounded in the top-predicted context but acknowledges the residual probability mass.

(Note: ESC-50 cat clips are out-of-distribution for the classifier — it was trained only on CatMeows' 3 contexts. The high brushing probability here is the classifier's best mapping of an unfamiliar context to its closest CatMeows class.)

---

## Why bridge instead of fine-tune?

NatureLM-audio is pretrained for general bioacoustics across many species. Replacing its Llama-3.1-8B head with a larger reasoner (GPT-OSS-20B) would require retraining the audio→LLM projection, which is expensive. Bridging is cheaper and modular:

- **NatureLM-audio** stays untouched → keeps its broad bioacoustic prior
- **Small classifier head** adds domain-specific grounding (cats) with ~200K trainable params
- **GPT-OSS-20B** does the heavy reasoning over both caption + context probabilities

On an 8 GB consumer GPU this works because GPT-OSS-20B is MoE: only ~3.6B params active per token, so `llama.cpp -ngl 12` with Q4_K_M fits the always-on backbone on the GPU while expert weights stream from system RAM. We measured **32 tok/s generation** on RTX 4070 Laptop.

---

## Results

Cat-context classifier on [CatMeows](https://zenodo.org/records/4008297), evaluated via **5-fold stratified cross-validation across all 276 clips**, hyperparameters tuned per-variant via grid sweep:

| Feature variant | 5-fold CV accuracy | Best hparams |
|---|---|---|
| **Hybrid: BEATs-stats + classical (2406-dim)** | **82.2%** | hidden=128, dropout=0.0, wd=1e-1 |
| BEATs-stats only (2304-dim, mean+std+max) | 81.5% | hidden=256, dropout=0.3, wd=1e-1 |
| BEATs-mean only (768-dim) | 76.5% | hidden=128, dropout=0.7, wd=1e-1 |
| Q-Former only (768-dim) | 73.9% | hidden=256, dropout=0.0, wd=1e-2 |
| Classical only (102-dim, MFCC+spectral+ZCR+RMS) | 72.5% | hidden=64, dropout=0.3, wd=1e-2 |

Chance = 33%. The Ludovico et al. 2020 paper reports 86% on the same dataset using hand-crafted features (MFCC + spectral centroid + ZCR + formants) + SVM, with full-dataset CV. Our **82.2% hybrid** uses learned BEATs features + classical acoustic features + a small MLP head — **within 4 points of the hand-crafted SVM baseline**, despite a meaningfully different model class.

### Stage-by-stage progression

| Stage | Accuracy | Lift |
|---|---|---|
| Single 201/75 split (initial, misleading) | 65.3% | baseline |
| 5-fold CV, BEATs-mean, default hparams | 76.5% | +11.2 (single-split was unlucky) |
| BEATs-stats pooling (mean+std+max) | 78.6% | +2.1 |
| BEATs-stats, tuned hparams | 81.5% | +2.9 (regularization is the unlock) |
| Hybrid with classical features | 82.2% | +0.7 (marginal but free) |

### Negative result: NatureLM-audio's Q-Former features perform *worse* than raw BEATs

Counterintuitive but reproducible: feeding the cat audio through NatureLM-audio's Q-Former (post-BEATs) and using the Q-Former output as classifier input gives **73.9%**, ~5-8 points lower than raw BEATs features. Our interpretation: NatureLM-audio's Q-Former is designed as a tight bottleneck (1 query token per 333ms window) optimized for downstream Llama text generation; that bottleneck throws away acoustic detail needed for fine-grained cat-context discrimination. **Statistical pooling on raw BEATs preserves more variability than the Q-Former bottleneck allows through.**

### Negative result: off-the-shelf data augmentation hurt

Time-stretch (0.9-1.1×) + pitch-shift (±2 semitones) + light noise augmentation dropped CV from 78.6% to 75.4% with the BEATs-stats default hparams. Pitch-shift in particular seems to move meows out of the discriminative acoustic range. SEM also went up from ±1.5% to ±2.3%, indicating augmentation introduced fold-to-fold variance. Documented in `scripts/extract_beats_stats_augmented.py` for transparency.

### Per-class behavior

Across folds, the most-confused pair is **brushing ↔ isolation** (both produce gentler, non-distressed meows). Food-context meows are most distinguishable (sharper onsets, more urgent envelope).

---

## Install

Prereqs:
- HuggingFace auth with access to `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated)
- `uv` for Python env management
- llama.cpp built with CUDA (we used build 7787+)
- ~30 GB disk for model checkpoints

```bash
git clone https://github.com/Scottcjn/naturelm-cats
cd naturelm-cats
uv sync --no-group gpu      # or "uv sync" if you want CUDA torch
```

Download required models:

```bash
# GPT-OSS-20B GGUF (~11 GB)
hf download unsloth/gpt-oss-20b-GGUF gpt-oss-20b-Q4_K_M.gguf \
  --local-dir ~/models/gpt-oss-20b-gguf

# NatureLM-audio head (~1.5 GB)
hf download EarthSpeciesProject/NatureLM-audio --local-dir ~/models/naturelm-audio

# Llama-3.1-8B base (NatureLM-audio's LoRA backbone, ~16 GB)
hf download meta-llama/Meta-Llama-3.1-8B-Instruct --exclude "original/*"

# CatMeows dataset (for training the classifier yourself, ~18 MB)
hf download oliveirabruno01/openfarm-catmeows --repo-type dataset \
  --local-dir ~/datasets/cats/openfarm-catmeows
```

---

## Run

Start the LLM server (terminal 1):

```bash
~/llama.cpp/build-cuda/bin/llama-server \
  --model ~/models/gpt-oss-20b-gguf/gpt-oss-20b-Q4_K_M.gguf \
  --host 127.0.0.1 --port 9002 \
  --n-gpu-layers 12 --ctx-size 4096 --threads 8
```

Run the bridge against any cat audio (terminal 2):

```bash
# Caption-only bridge (no trained classifier)
CUDA_VISIBLE_DEVICES="" uv run python scripts/bridge_caption.py path/to/cat.wav

# Bridge v2: classifier-grounded reasoning (default)
CUDA_VISIBLE_DEVICES="" uv run python scripts/bridge_with_context.py path/to/cat.wav

# Bridge v3: multi-query — runs 3 NatureLM queries, flags disagreements
CUDA_VISIBLE_DEVICES="" uv run python scripts/bridge_multiquery.py path/to/cat.wav

# Batch over many clips at once (writes /tmp/cat_batch_{results.jsonl,report.md})
CUDA_VISIBLE_DEVICES="" uv run python scripts/batch_bridge.py --source mixed --n-each 5

# Probe which NatureLM queries produce informative captions
CUDA_VISIBLE_DEVICES="" uv run python scripts/probe_naturelm_queries.py
```

`CUDA_VISIBLE_DEVICES=""` forces NatureLM-audio to CPU so it doesn't fight llama-server for VRAM. ~60s cold start (Llama-3.1-8B loads into 16 GB of system RAM), <10s warm.

---

## Train the classifier yourself

```bash
# 1. Extract BEATs-stats + classical features for all 276 CatMeows clips
CUDA_VISIBLE_DEVICES="" uv run python scripts/extract_beats_stats_features.py
uv run python scripts/extract_classical_features.py
# Build the hybrid concat (saved as hybrid_features_{train,test}.npy)
uv run python -c "
import numpy as np
from pathlib import Path
D = Path.home() / 'datasets/cats'
for split in ['train', 'test']:
    h = np.concatenate([np.load(D/f'stats_features_{split}.npy'),
                        np.load(D/f'classical_features_{split}.npy')], axis=1)
    np.save(D/f'hybrid_features_{split}.npy', h.astype(np.float32))
"

# 2. Train final classifier on ALL data (no held-out — use CV results for accuracy estimate)
uv run python scripts/train_context_classifier.py --variant hybrid --all

# 3. (Optional) Confirm via 5-fold CV + full hyperparam sweep across all variants
uv run python scripts/cross_validate.py
uv run python scripts/sweep_features.py
```

Outputs `~/datasets/cats/context_classifier_hybrid_all.pt` (used by `bridge_with_context.py`) plus per-variant `cv_results.json` and `cv_results_augmented.json`.

### Other variants (kept for comparison + transparency on negative results)

```bash
# BEATs-mean only — 76.5% CV (simpler baseline)
CUDA_VISIBLE_DEVICES="" uv run python scripts/extract_beats_features.py
uv run python scripts/train_context_classifier.py --variant mean --all

# BEATs-stats only — 81.5% CV (no classical features)
uv run python scripts/train_context_classifier.py --variant stats --all

# Q-Former features — 73.9% CV (negative result, documents bottleneck)
CUDA_VISIBLE_DEVICES="" uv run python scripts/extract_qformer_features.py

# Augmented BEATs-stats — 75.4% CV (negative result, off-the-shelf aug hurt)
CUDA_VISIBLE_DEVICES="" uv run python scripts/extract_beats_stats_augmented.py
uv run python scripts/cross_validate_augmented.py
```

---

## Files added on top of upstream

```
scripts/
├── bridge_caption.py                    v1 bridge: NatureLM caption → GPT-OSS-20B
├── bridge_with_context.py               v2 bridge: + hybrid classifier (82.2% CV)
├── bridge_multiquery.py                 v3 bridge: 3 NatureLM queries + classifier
├── batch_bridge.py                      batch-process many clips, save JSONL + Markdown
├── probe_naturelm_queries.py            evaluate which NatureLM queries are informative
├── extract_beats_features.py            BEATs mean-pool features (baseline)
├── extract_beats_stats_features.py      BEATs mean+std+max stats features
├── extract_classical_features.py        102-dim MFCC+spectral+ZCR+RMS via librosa
├── extract_qformer_features.py          Q-Former features (negative-result experiment)
├── extract_beats_stats_augmented.py     5×-augmented BEATs-stats (negative-result experiment)
├── train_context_classifier.py          MLP head training, --variant {mean|stats|hybrid}, --all
├── cross_validate.py                    5-fold stratified CV across {mean, stats, Q-Former}
├── cross_validate_augmented.py          5-fold group-aware CV with augmented training
├── sweep_features.py                    80-config hyperparam sweep across all 5 variants
└── smoke_test_gpt_oss.py                standalone GPT-OSS-20B load test (transformers path)
```

Nothing in `NatureLM/` is modified — this is purely additive on top of upstream `earthspecies/NatureLM-audio`.

---

## Licensing

- **This repo's added code** (under `scripts/`): Apache 2.0, same as upstream
- **NatureLM-audio model weights** (used at runtime): CC-BY-NC-SA 4.0 per [EarthSpeciesProject/NatureLM-audio](https://huggingface.co/EarthSpeciesProject/NatureLM-audio) — our derivative trained classifier inherits the NC restriction
- **CatMeows dataset**: CC-BY-4.0 per [Zenodo record 4008297](https://zenodo.org/records/4008297)
- **GPT-OSS-20B**: Apache 2.0 per OpenAI release
- **Llama 3.1**: Llama 3.1 Community License (separate, gated access on HF)

If you want commercial use of the trained cat classifier, contact Earth Species Project regarding their NC-licensed weights — the classifier head itself is small enough (~200K params) to retrain on different upstream features.

---

## Limitations

- **3 contexts only** (brushing / isolation / waiting-for-food). Real cats produce many more vocalization types; the classifier will force-fit out-of-distribution audio into one of these three. Use the softmax confidence as your uncertainty signal.
- **82.2% CV accuracy** (hybrid BEATs+classical) vs Ludovico et al. 2020's 86% — within 4 points. Off-the-shelf data augmentation hurt rather than helped. Closing the remaining gap likely needs (a) more data, (b) per-cat personalization, or (c) a learned attention pool over BEATs time-steps instead of statistical pooling.
- **No per-cat personalization**: the classifier was trained on 21 cats from CatMeows. Individual cats have idiosyncratic dialects; a per-cat embedding or fine-tuning on a few labeled clips of your specific cat would help substantially.
- **No reverse direction**: we listen to the cat, we don't generate audio the cat would recognize. Cat-directed speech synthesis is a separate, harder problem.

---

## Acknowledgements

This project is a soft fork of [earthspecies/NatureLM-audio](https://github.com/earthspecies/NatureLM-audio). The audio-language foundation model, BEATs encoder fine-tune, and the entire bioacoustic pretraining stack are their work.

- **NatureLM-audio**: Robinson et al., *NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics*, ICLR 2025. [paper](https://openreview.net/forum?id=hJVdwBpWjt)
- **CatMeows dataset**: Ludovico et al., *CatMeows: A Publicly-Available Dataset of Cat Vocalizations*, Animals 2020. [DOI 10.5281/zenodo.4008297](https://zenodo.org/records/4008297)
- **GPT-OSS-20B**: OpenAI open-weights release
- **Earth Species Project's mission**: decoding non-human communication for science and conservation. The bridge architecture here is a small extension to consumer hardware; their foundation models do the actual heavy lifting.
