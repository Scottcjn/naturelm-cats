"""Batch process many clips through bridge v3 (multi-query NatureLM + classifier).

Per clip, captures:
  - cat-context classifier probabilities
  - NatureLM-audio answers to 3 queries (acoustic / emotion / sentence)
  - GPT-OSS-20B reasoning (which can flag caption-vs-classifier disagreements)
  - ground-truth (if from CatMeows-test)

Writes /tmp/cat_batch_v3_results.jsonl + /tmp/cat_batch_v3_report.md.

~15-20s/clip warm.  For 20 clips, plan ~5-6 minutes after NatureLM loads.
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

QUERIES = {
    "acoustic": "Describe the acoustic characteristics of this vocalization: pitch, rhythm, duration, and intensity.",
    "emotion": "Describe the emotional tone of this animal vocalization. Is the animal calm, distressed, or excited?",
    "sentence": "If this vocalization were a sentence the animal was saying, what would it be?",
}


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


def _classical_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
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


def classify_context(audio: np.ndarray, beats, classifier, mu, sd, label_names):
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    with torch.inference_mode():
        wav = torch.from_numpy(audio).unsqueeze(0)
        feats, _ = beats(wav)
        x = feats.squeeze(0)
        beats_pool = torch.cat([x.mean(0), x.std(0), x.max(0).values], dim=0)
        cls_pool = torch.from_numpy(_classical_features(audio))
        pooled = torch.cat([beats_pool, cls_pool], dim=0).unsqueeze(0)
        pooled = (pooled - mu) / sd
        probs = torch.softmax(classifier(pooled), dim=-1).squeeze(0).cpu().numpy()
    ranked = sorted([(label_names[i], float(probs[i])) for i in range(len(probs))], key=lambda x: -x[1])
    return {"probabilities": dict(ranked), "top_label": ranked[0][0], "top_prob": ranked[0][1]}


def preprocess_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (unpadded_16k_audio_for_classifier, padded_10s_audio_for_naturelm).

    BUG FIX: the classifier was trained on original-length audio (no padding).
    Padding to 10s injects ~8 seconds of silence into short clips, distorts the
    mean/std/max statistics, and biases the classifier toward 'brushing'.
    NatureLM-audio DOES need padding to 10s for its window logic.
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if sr != SAMPLE_RATE:
        audio = resampy.resample(audio.astype(np.float32), sr, SAMPLE_RATE)
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

    target = SAMPLE_RATE * 10
    if len(audio) < target:
        padded = np.pad(audio, (0, target - len(audio)))
    else:
        padded = audio[:target]
        audio = audio[:target]
    return audio, padded


def multi_caption(pipeline, audio_10s: np.ndarray) -> dict[str, str]:
    answers = {}
    for name, q in QUERIES.items():
        out = pipeline([audio_10s], [q], window_length_seconds=10.0, hop_length_seconds=10.0,
                      input_sample_rate=SAMPLE_RATE)
        a = (out[0] if isinstance(out, list) else str(out)).strip()
        if a.startswith("#") and "#:" in a:
            a = a.split("#:", 1)[1].strip()
        answers[name] = a
    return answers


def reason(captions: dict, ctx: dict, ground_truth: str | None) -> str:
    """ground_truth is NOT passed to the model — we keep it on the record for scoring only.

    Earlier versions of this script leaked GT into the prompt with a 'for evaluation only'
    disclaimer. The model used it anyway, invalidating any 'GPT-OSS-20B caught the
    classifier's error' interpretation. Fixed: GT now stays in the JSONL/MD report only.
    """
    probs_pct = {k: f"{v:.1%}" for k, v in ctx["probabilities"].items()}
    messages = [
        {"role": "system", "content": (
            "You are a bioacoustics expert. The user gives you three different captions of "
            "the same audio clip (different question framings to NatureLM-audio) and a "
            "cat-context classifier's probabilities (CatMeows dataset, 82.2% 5-fold CV). "
            "Reason carefully and surface any caption-vs-classifier disagreements explicitly. "
            "If captions and classifier disagree, weigh the evidence and commit to a "
            "best-guess interpretation while noting your uncertainty. "
            "Keep your answer under 200 words."
        )},
        {"role": "user", "content": (
            f"Caption (acoustic): {captions['acoustic']!r}\n"
            f"Caption (emotion): {captions['emotion']!r}\n"
            f"Caption (sentence): {captions['sentence']!r}\n\n"
            f"Cat classifier: top={ctx['top_label']} ({ctx['top_prob']:.1%}), full={probs_pct}\n\n"
            f"What is this cat most likely communicating?"
        )},
    ]
    r = requests.post(
        f"{LLAMA_SERVER}/v1/chat/completions",
        json={"model": "gpt-oss-20b", "messages": messages, "max_tokens": 1000, "temperature": 0.3},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def collect_clips(n_esc50: int, n_catmeows: int, rng: random.Random):
    clips = []
    if n_esc50 > 0:
        with open(ESC50_META) as f:
            rows = [r for r in csv.DictReader(f) if r["category"] == "cat"]
        rng.shuffle(rows)
        for row in rows[:n_esc50]:
            audio, sr = sf.read(str(ESC50_AUDIO / row["filename"]))
            clips.append({"source": "ESC-50", "clip_id": row["filename"],
                          "ground_truth": None, "audio": audio.astype(np.float32), "sr": sr})
    if n_catmeows > 0:
        df = pq.read_table(CATMEOWS_TEST).to_pandas()
        idxs = list(range(len(df)))
        rng.shuffle(idxs)
        for i in idxs[:n_catmeows]:
            row = df.iloc[i]
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
            clips.append({"source": "CatMeows", "clip_id": row["audio_filename"],
                          "ground_truth": row["context"], "cat_id": row.get("cat_id"),
                          "breed": row.get("breed"),
                          "audio": audio.astype(np.float32), "sr": sr})
    return clips


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-esc50", type=int, default=10)
    ap.add_argument("--n-catmeows", type=int, default=10)
    ap.add_argument("--out-prefix", default="/tmp/cat_batch_v3")
    args = ap.parse_args()

    try:
        requests.get(f"{LLAMA_SERVER}/health", timeout=3).raise_for_status()
    except Exception as e:
        print(f"FAIL: llama-server unreachable ({e})")
        return 1

    print("[setup] Loading BEATs + classifier...", file=sys.stderr)
    t0 = time.time()
    beats = build_beats()
    ckpt = torch.load(CLASSIFIER_PATH, weights_only=False)
    cfg = ckpt["config"]
    classifier = ContextHead(cfg["in_dim"], cfg["hidden"], cfg["n_classes"], cfg["dropout"])
    classifier.load_state_dict(ckpt["state_dict"]); classifier.eval()
    mu, sd, label_names = ckpt["feature_mu"], ckpt["feature_sd"], ckpt["label_names"]
    print(f"[setup] Loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    print("[setup] Loading NatureLM-audio (CPU)...", file=sys.stderr)
    t0 = time.time()
    from NatureLM.infer import Pipeline
    pipeline = Pipeline()
    print(f"[setup] NatureLM loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    rng = random.Random(SEED)
    clips = collect_clips(args.n_esc50, args.n_catmeows, rng)
    print(f"[setup] {len(clips)} clips queued", file=sys.stderr)

    records = []
    for i, clip in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] {clip['source']} {clip['clip_id']}", file=sys.stderr)
        t = time.time()

        audio_cls, audio_naturelm = preprocess_audio(clip["audio"], clip["sr"])
        ctx = classify_context(audio_cls, beats, classifier, mu, sd, label_names)
        print(f"  classifier: top={ctx['top_label']} ({ctx['top_prob']:.1%})", file=sys.stderr)

        captions = multi_caption(pipeline, audio_naturelm)
        print(f"  acoustic: {captions['acoustic']!r}", file=sys.stderr)
        print(f"  emotion:  {captions['emotion']!r}", file=sys.stderr)
        print(f"  sentence: {captions['sentence']!r}", file=sys.stderr)

        reasoning = reason(captions, ctx, clip["ground_truth"])
        records.append({
            "source": clip["source"], "clip_id": clip["clip_id"],
            "ground_truth": clip["ground_truth"],
            "classifier": ctx, "captions": captions, "reasoning": reasoning,
            "elapsed_s": round(time.time() - t, 1),
        })

    jsonl = Path(f"{args.out_prefix}_results.jsonl")
    with jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    md = Path(f"{args.out_prefix}_report.md")
    with md.open("w") as f:
        f.write(f"# Cat batch v3 (multi-query) — {len(records)} clips\n\n")
        f.write("## Summary\n\n")
        f.write("| # | Source | Clip | Ground truth | Pred | Conf | Match |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(records, 1):
            gt = r['ground_truth'] or '(OOD)'
            match = "✓" if r['ground_truth'] and r['classifier']['top_label'] == r['ground_truth'] else (
                    "✗" if r['ground_truth'] else "—")
            f.write(f"| {i} | {r['source']} | `{r['clip_id']}` | {gt} | "
                    f"{r['classifier']['top_label']} | {r['classifier']['top_prob']:.0%} | {match} |\n")
        f.write("\n## Per-clip detail\n\n")
        for i, r in enumerate(records, 1):
            f.write(f"### {i}. `{r['clip_id']}` ({r['source']})\n\n")
            if r['ground_truth']:
                f.write(f"**Ground truth:** {r['ground_truth']}\n\n")
            probs = r['classifier']['probabilities']
            f.write("**Classifier:**  " + ", ".join(f"{k}: {v:.1%}" for k, v in probs.items()) + "\n\n")
            f.write("**NatureLM captions:**\n\n")
            for q, a in r['captions'].items():
                f.write(f"- *{q}:* `{a}`\n")
            f.write("\n**GPT-OSS-20B:**\n\n")
            for line in r['reasoning'].splitlines():
                f.write(f"> {line}\n")
            f.write(f"\n_elapsed {r['elapsed_s']}s_\n\n---\n\n")

    print(f"\nWrote {jsonl}\nWrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
