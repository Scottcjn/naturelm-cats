"""Bridge v3: multi-query NatureLM + hybrid classifier + GPT-OSS-20B.

Hits NatureLM-audio with three complementary queries per clip:
    1. Acoustic characteristics  → richest descriptive caption
    2. Emotional tone            → independent emotion signal (often disagrees with classifier!)
    3. If-a-sentence translation → generative interpretation, distinct per clip

Bundles all three caption signals + classifier probabilities into GPT-OSS-20B's prompt.

This is the richest version of the bridge. ~3× NatureLM inference cost (~15s/clip warm)
but the multi-query bundle gives GPT-OSS-20B much more signal to reason over than the
single "Caption the audio" query which often returns "None".

Run:
    cd /home/scott/naturelm-cats
    CUDA_VISIBLE_DEVICES="" uv run python scripts/bridge_multiquery.py <audio.wav>
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import librosa
import numpy as np
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
SAMPLE_RATE = 16000
CV_ACCURACY = 0.822

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


def classify_context(audio: np.ndarray, beats, classifier, mu, sd, label_names) -> dict:
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if (len(audio) == 0):
        raise ValueError("empty audio")
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
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


def multi_query_caption(audio_path: Path) -> dict[str, str]:
    """Run all 3 NatureLM queries against audio_path. Returns dict keyed by query name."""
    from NatureLM.infer import Pipeline

    print("[caption] Loading NatureLM-audio (CPU)...", file=sys.stderr)
    t0 = time.time()
    pipeline = Pipeline()
    print(f"[caption] Loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    # Load + preprocess audio once
    audio, sr = sf.read(str(audio_path))
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if sr != 16000:
        audio = resampy.resample(audio.astype(np.float32), sr, 16000)
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    target = 16000 * 10
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    answers = {}
    for name, q in QUERIES.items():
        t = time.time()
        out = pipeline([audio], [q], window_length_seconds=10.0, hop_length_seconds=10.0,
                      input_sample_rate=16000)
        a = (out[0] if isinstance(out, list) else str(out)).strip()
        if a.startswith("#") and "#:" in a:
            a = a.split("#:", 1)[1].strip()
        answers[name] = a
        print(f"[caption.{name}] {a!r}  ({time.time()-t:.1f}s)", file=sys.stderr)

    return answers


def reason_with_gpt_oss(captions: dict[str, str], ctx: dict) -> str:
    probs_pct = {k: f"{v:.1%}" for k, v in ctx["probabilities"].items()}
    messages = [
        {"role": "system", "content": (
            "You are a bioacoustics expert. The user gives you three different captions of "
            "the SAME audio clip (from different question framings of a bioacoustic foundation "
            "model), plus probabilities from a cat-context classifier trained on the CatMeows "
            f"dataset (3 contexts, 5-fold CV accuracy {CV_ACCURACY:.1%}). Reason carefully. "
            "Pay special attention to disagreements between the captions and the classifier — "
            "e.g., if the emotional-tone caption says 'distressed' but the classifier says "
            "'brushing' with high confidence, surface that mismatch explicitly. "
            "Keep your answer under 200 words."
        )},
        {"role": "user", "content": (
            f"Caption (acoustic characteristics): {captions['acoustic']!r}\n"
            f"Caption (emotional tone): {captions['emotion']!r}\n"
            f"Caption (if-a-sentence translation): {captions['sentence']!r}\n\n"
            f"Cat-context classifier: top={ctx['top_label']} ({ctx['top_prob']:.1%}), full={probs_pct}\n\n"
            f"What is this cat most likely communicating? Flag any caption-vs-classifier disagreement."
        )},
    ]
    r = requests.post(
        f"{LLAMA_SERVER}/v1/chat/completions",
        json={"model": "gpt-oss-20b", "messages": messages, "max_tokens": 1200, "temperature": 0.3},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    args = p.parse_args()

    if not args.audio.exists():
        print(f"FAIL: {args.audio} not found")
        return 1
    try:
        requests.get(f"{LLAMA_SERVER}/health", timeout=3).raise_for_status()
    except Exception as e:
        print(f"FAIL: llama-server unreachable at {LLAMA_SERVER} ({e})")
        return 1

    print("[ctx] Loading BEATs + classifier...", file=sys.stderr)
    t0 = time.time()
    beats = build_beats()
    ckpt = torch.load(CLASSIFIER_PATH, weights_only=False)
    cfg = ckpt["config"]
    classifier = ContextHead(cfg["in_dim"], cfg["hidden"], cfg["n_classes"], cfg["dropout"])
    classifier.load_state_dict(ckpt["state_dict"]); classifier.eval()
    mu, sd = ckpt["feature_mu"], ckpt["feature_sd"]
    label_names = ckpt["label_names"]
    print(f"[ctx] Loaded in {time.time()-t0:.1f}s. CV accuracy: {CV_ACCURACY:.1%}", file=sys.stderr)

    # Classifier on raw audio
    audio_raw, sr = sf.read(str(args.audio))
    if sr != SAMPLE_RATE:
        if audio_raw.ndim == 2:
            audio_raw = audio_raw.mean(axis=int(np.argmin(audio_raw.shape)))
        audio_raw = resampy.resample(audio_raw.astype(np.float32), sr, SAMPLE_RATE)
    ctx = classify_context(audio_raw, beats, classifier, mu, sd, label_names)

    print(f"\n=== Cat-context classifier ===\nTop: {ctx['top_label']} ({ctx['top_prob']:.1%})")
    for k, v in ctx["probabilities"].items():
        print(f"  {k}: {v:.1%}")

    captions = multi_query_caption(args.audio)
    print(f"\n=== NatureLM multi-query ===")
    for q, a in captions.items():
        print(f"  [{q}] {a}")

    answer = reason_with_gpt_oss(captions, ctx)
    print(f"\n=== GPT-OSS-20B reasoning (with multi-query) ===\n{answer}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
