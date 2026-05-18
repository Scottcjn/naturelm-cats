"""Batch-process cat audio through the bridge, save results for review.

For each clip, captures:
    - cat-context classifier probabilities
    - NatureLM-audio caption (bioacoustic generic)
    - GPT-OSS-20B structured reasoning
    - ground-truth label (if provided)

Writes:
    /tmp/cat_batch_results.jsonl   — one JSON record per clip
    /tmp/cat_batch_report.md       — human-readable markdown summary

Usage:
    uv run python scripts/batch_bridge.py --source esc50 --n 10
    uv run python scripts/batch_bridge.py --source catmeows-test --n 10
    uv run python scripts/batch_bridge.py --source mixed --n-each 5
"""
import argparse
import csv
import io
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import librosa
import numpy as np
import pyarrow.parquet as pq
import requests
import resampy
import soundfile as sf
import torch
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, "/home/scott/naturelm-cats")
from NatureLM.models.beats.BEATs import BEATs, BEATsConfig

LLAMA_SERVER = "http://127.0.0.1:9002"
CLASSIFIER_PATH = Path("/home/scott/datasets/cats/context_classifier_hybrid_all.pt")
NLM_CKPT_PATH = Path("/home/scott/models/naturelm-audio/model.safetensors")
ESC50_AUDIO = Path("/home/scott/ESC-50/audio")
ESC50_META = Path("/home/scott/ESC-50/meta/esc50.csv")
CATMEOWS_TEST = Path("/home/scott/datasets/cats/openfarm-catmeows/data/test-00000-of-00001.parquet")
SAMPLE_RATE = 16000
SEED = 13


# ───────────────────────── Audio side (shared with bridge_with_context) ─────────────────────────

def build_beats():
    cfg_dict = {
        "input_patch_size": 16, "embed_dim": 512, "conv_bias": False,
        "encoder_layers": 12, "encoder_embed_dim": 768, "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12, "activation_fn": "gelu",
        "layer_wise_gradient_decay_ratio": 0.6, "layer_norm_first": False,
        "deep_norm": True, "dropout": 0.0, "attention_dropout": 0.0,
        "activation_dropout": 0.0, "encoder_layerdrop": 0.05, "dropout_input": 0.0,
        "conv_pos": 128, "conv_pos_groups": 16, "relative_position_embedding": True,
        "num_buckets": 320, "max_distance": 800, "gru_rel_pos": True,
        "finetuned_model": True, "predictor_dropout": 0.0, "predictor_class": 527,
    }
    beats = BEATs(cfg=BEATsConfig(cfg_dict))
    full = load_file(str(NLM_CKPT_PATH))
    beats_state = {k[len("beats."):]: v for k, v in full.items() if k.startswith("beats.")}
    beats.load_state_dict(beats_state, strict=False)
    beats.eval()
    return beats


class ContextHead(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)


def classical_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    n_fft, hop = 1024, 256
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop)
    def pool(x): return np.concatenate([x.mean(axis=-1), x.std(axis=-1)])
    return np.concatenate([pool(mfcc), pool(delta), pool(delta2), pool(zcr),
                           pool(centroid), pool(bw), pool(rolloff),
                           pool(contrast), pool(rms)]).astype(np.float32)


def classify_context(audio_np: np.ndarray, beats, classifier, mu, sd, label_names):
    with torch.inference_mode():
        wav = torch.from_numpy(audio_np).unsqueeze(0)
        feats, _ = beats(wav)
        x = feats.squeeze(0)
        beats_pool = torch.cat([x.mean(0), x.std(0), x.max(0).values], dim=0)
        cls_pool = torch.from_numpy(classical_features(audio_np))
        pooled = torch.cat([beats_pool, cls_pool], dim=0).unsqueeze(0)
        pooled = (pooled - mu) / sd
        probs = torch.softmax(classifier(pooled), dim=-1).squeeze(0).cpu().numpy()
    ranked = sorted([(label_names[i], float(probs[i])) for i in range(len(probs))], key=lambda x: -x[1])
    return {"probabilities": dict(ranked), "top_label": ranked[0][0], "top_prob": ranked[0][1]}


def reason_with_gpt_oss(caption: str, ctx_result: dict, ground_truth: str | None) -> str:
    probs_pct = {k: f"{v:.1%}" for k, v in ctx_result["probabilities"].items()}
    gt_line = f"GROUND TRUTH (for evaluation only, do not bias toward this): {ground_truth}\n" if ground_truth else ""
    messages = [
        {"role": "system", "content": (
            "You are a bioacoustics expert. The user gives you a generic audio caption "
            "from a bioacoustic foundation model and probabilities from a cat-context classifier "
            "trained on the CatMeows dataset (3 contexts: brushing, isolation, waiting-for-food, "
            "5-fold CV accuracy 82.2%). Reason carefully. Flag uncertainty when classifier top "
            "probability is below 50%. Note any caption-vs-classifier disagreement explicitly. "
            "Keep your answer under 150 words."
        )},
        {"role": "user", "content": (
            f"Generic caption: {caption!r}\n\n"
            f"Cat classifier: top={ctx_result['top_label']} ({ctx_result['top_prob']:.1%}), full={probs_pct}\n\n"
            f"{gt_line}"
            f"What is this cat most likely communicating, and how confident are you?"
        )},
    ]
    r = requests.post(
        f"{LLAMA_SERVER}/v1/chat/completions",
        json={"model": "gpt-oss-20b", "messages": messages, "max_tokens": 1000, "temperature": 0.3},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ───────────────────────── Clip source selection ─────────────────────────

def collect_clips(source: str, n: int, n_each: int, rng: random.Random) -> list[dict]:
    """Return list of dicts: {audio_np, sample_rate, source, clip_id, ground_truth?}"""
    clips = []

    if source in ("esc50", "mixed"):
        with open(ESC50_META) as f:
            cat_rows = [r for r in csv.DictReader(f) if r["category"] == "cat"]
        rng.shuffle(cat_rows)
        take = n_each if source == "mixed" else n
        for row in cat_rows[:take]:
            audio, sr = sf.read(str(ESC50_AUDIO / row["filename"]))
            clips.append({
                "audio_np": audio.astype(np.float32),
                "sample_rate": sr,
                "source": "ESC-50",
                "clip_id": row["filename"],
                "ground_truth": "cat (context unknown — sourced from YouTube)",
            })

    if source in ("catmeows-test", "mixed"):
        df = pq.read_table(CATMEOWS_TEST).to_pandas()
        idxs = list(range(len(df)))
        rng.shuffle(idxs)
        take = n_each if source == "mixed" else n
        for i in idxs[:take]:
            row = df.iloc[i]
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            clips.append({
                "audio_np": audio.astype(np.float32),
                "sample_rate": sr,
                "source": "CatMeows-test",
                "clip_id": row["audio_filename"],
                "ground_truth": row["context"],
                "cat_id": row.get("cat_id", "?"),
                "breed": row.get("breed", "?"),
            })

    return clips


# ───────────────────────── NatureLM caption (one-shot per clip) ─────────────────────────

def caption_clip(pipeline, clip: dict, query: str) -> str:
    audio = clip["audio_np"]
    sr = clip["sample_rate"]
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    target_len = 16000 * 10
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    out = pipeline([audio], [query], window_length_seconds=10.0, hop_length_seconds=10.0,
                   input_sample_rate=16000)
    return out[0] if isinstance(out, list) else str(out)


def preprocess_for_classifier(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if sr != SAMPLE_RATE:
        audio = resampy.resample(audio.astype(np.float32), sr, SAMPLE_RATE)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ───────────────────────── Main ─────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["esc50", "catmeows-test", "mixed"], default="mixed")
    ap.add_argument("--n", type=int, default=10, help="Total clips for single-source modes")
    ap.add_argument("--n-each", type=int, default=5, help="Per-source for 'mixed' mode")
    ap.add_argument("--query", default="Caption the audio. What animal vocalization is this, if any?")
    ap.add_argument("--out-prefix", default="/tmp/cat_batch")
    args = ap.parse_args()

    try:
        requests.get(f"{LLAMA_SERVER}/health", timeout=3).raise_for_status()
    except Exception as e:
        print(f"FAIL: llama-server unreachable at {LLAMA_SERVER} ({e})")
        return 1

    print(f"[setup] Loading BEATs + classifier...", file=sys.stderr)
    t0 = time.time()
    beats = build_beats()
    ckpt = torch.load(CLASSIFIER_PATH, weights_only=False)
    cfg = ckpt["config"]
    classifier = ContextHead(cfg["in_dim"], cfg["hidden"], cfg["n_classes"], cfg["dropout"])
    classifier.load_state_dict(ckpt["state_dict"]); classifier.eval()
    mu, sd = ckpt["feature_mu"], ckpt["feature_sd"]
    label_names = ckpt["label_names"]
    print(f"[setup] Classifier loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    print(f"[setup] Loading NatureLM-audio (CPU)...", file=sys.stderr)
    t0 = time.time()
    from NatureLM.infer import Pipeline
    pipeline = Pipeline()
    print(f"[setup] NatureLM loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    rng = random.Random(SEED)
    clips = collect_clips(args.source, args.n, args.n_each, rng)
    print(f"[setup] {len(clips)} clips queued from {args.source}", file=sys.stderr)

    records = []
    for i, clip in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] {clip['source']} {clip['clip_id']}", file=sys.stderr)
        t = time.time()
        # Audio preprocessing
        audio_norm = preprocess_for_classifier(clip["audio_np"], clip["sample_rate"])

        ctx = classify_context(audio_norm, beats, classifier, mu, sd, label_names)
        print(f"  classifier: top={ctx['top_label']} ({ctx['top_prob']:.1%})", file=sys.stderr)

        caption = caption_clip(pipeline, clip, args.query)
        print(f"  caption: {caption!r}", file=sys.stderr)

        reasoning = reason_with_gpt_oss(caption, ctx, clip.get("ground_truth"))
        print(f"  reasoning: {reasoning[:120]}...", file=sys.stderr)

        record = {
            "source": clip["source"], "clip_id": clip["clip_id"],
            "ground_truth": clip.get("ground_truth"),
            "cat_id": clip.get("cat_id"), "breed": clip.get("breed"),
            "classifier": ctx, "caption": caption.strip(), "reasoning": reasoning.strip(),
            "elapsed_s": round(time.time() - t, 1),
        }
        records.append(record)

    # Write outputs
    jsonl_path = Path(f"{args.out_prefix}_results.jsonl")
    with jsonl_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {jsonl_path}")

    md_path = Path(f"{args.out_prefix}_report.md")
    with md_path.open("w") as f:
        f.write(f"# Cat batch run — {len(records)} clips\n\n")
        f.write(f"Source: `{args.source}`  ·  Classifier: hybrid (82.2% CV)\n\n")

        # Quick summary table
        f.write("## Summary\n\n")
        f.write("| # | Source | Clip | Ground truth | Classifier top | Confidence |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, r in enumerate(records, 1):
            f.write(f"| {i} | {r['source']} | `{r['clip_id']}` | {r['ground_truth']} | "
                    f"{r['classifier']['top_label']} | {r['classifier']['top_prob']:.1%} |\n")

        f.write("\n## Per-clip detail\n\n")
        for i, r in enumerate(records, 1):
            f.write(f"### {i}. `{r['clip_id']}` ({r['source']})\n\n")
            if r.get("ground_truth"):
                f.write(f"**Ground truth:** {r['ground_truth']}")
                if r.get("cat_id"):
                    f.write(f"  (cat_id={r['cat_id']}, breed={r.get('breed')})")
                f.write("\n\n")
            probs = r["classifier"]["probabilities"]
            f.write("**Classifier:**  ")
            f.write(", ".join(f"{k}: {v:.1%}" for k, v in probs.items()) + "\n\n")
            f.write(f"**Caption:** `{r['caption']}`\n\n")
            f.write("**GPT-OSS-20B reasoning:**\n\n")
            for line in r["reasoning"].splitlines():
                f.write(f"> {line}\n")
            f.write(f"\n_elapsed {r['elapsed_s']}s_\n\n---\n\n")

    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
