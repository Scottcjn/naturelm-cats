"""Probe NatureLM-audio with multiple query framings per clip.

Goal: find queries that produce informative captions instead of "None"
or generic "Domestic cats vocalizing". The default query in our bridge
is "Caption the audio. What animal vocalization is this, if any?" —
which is often empty.

Tests N clips × 6 queries each.
"""
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pyarrow.parquet as pq
import resampy
import soundfile as sf

sys.path.insert(0, "/home/scott/naturelm-cats")

ESC50_AUDIO = Path("/home/scott/ESC-50/audio")
CATMEOWS_TEST = Path("/home/scott/datasets/cats/openfarm-catmeows/data/test-00000-of-00001.parquet")

QUERIES = [
    # Current default (baseline)
    "Caption the audio. What animal vocalization is this, if any?",
    # Emotion-focused
    "Describe the emotional tone of this animal vocalization. Is the animal calm, distressed, or excited?",
    # Acoustic-focused
    "Describe the acoustic characteristics of this vocalization: pitch, rhythm, duration, and intensity.",
    # Species + behavior
    "What species made this sound, and what behavioral context does the vocalization suggest?",
    # Story / translation
    "If this vocalization were a sentence the animal was saying, what would it be?",
    # Bioacoustic-specific
    "Identify the vocalization type (meow, purr, hiss, growl, trill, chirp) and any context cues.",
]


def load_audio_16k_10s(path_or_bytes) -> np.ndarray:
    if isinstance(path_or_bytes, (str, Path)):
        audio, sr = sf.read(str(path_or_bytes))
    else:
        audio, sr = sf.read(io.BytesIO(path_or_bytes))
    if audio.ndim == 2:
        audio = audio.mean(axis=int(np.argmin(audio.shape)))
    if sr != 16000:
        audio = resampy.resample(audio.astype(np.float32), sr, 16000)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    target = 16000 * 10
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return audio


def main() -> int:
    from NatureLM.infer import Pipeline

    print("[setup] Loading NatureLM-audio (CPU)...", file=sys.stderr)
    t0 = time.time()
    pipeline = Pipeline()
    print(f"[setup] Loaded in {time.time()-t0:.1f}s\n", file=sys.stderr)

    # 3 in-distribution (CatMeows-test, one per class) + 2 OOD (ESC-50)
    test_df = pq.read_table(CATMEOWS_TEST).to_pandas()
    clips = []
    for ctx in ["brushing", "isolation_unfamiliar_environment", "waiting_for_food"]:
        row = test_df[test_df["context"] == ctx].iloc[0]
        clips.append({
            "audio": load_audio_16k_10s(row["audio"]["bytes"]),
            "label": f"CatMeows / {ctx}",
            "id": row["audio_filename"],
        })
    for fn in ["1-34094-A-5.wav", "5-259169-A-5.wav"]:
        clips.append({
            "audio": load_audio_16k_10s(ESC50_AUDIO / fn),
            "label": "ESC-50 / unknown context",
            "id": fn,
        })

    results = []
    for clip in clips:
        print(f"\n=== {clip['label']} — {clip['id']} ===")
        clip_results = {"clip": clip['id'], "label": clip['label'], "answers": {}}
        for q in QUERIES:
            t = time.time()
            out = pipeline([clip["audio"]], [q], window_length_seconds=10.0,
                          hop_length_seconds=10.0, input_sample_rate=16000)
            answer = (out[0] if isinstance(out, list) else str(out)).strip()
            # Strip the "#0.00s - 10.00s#: " timestamp prefix to focus on content
            if answer.startswith("#") and "#:" in answer:
                answer = answer.split("#:", 1)[1].strip()
            elapsed = time.time() - t
            short_q = q[:50] + "..." if len(q) > 50 else q
            print(f"  Q: {short_q}")
            print(f"  A: {answer!r}  ({elapsed:.1f}s)")
            clip_results["answers"][q] = answer
        results.append(clip_results)

    # Aggregate analysis: which queries produced non-empty / non-"None" answers?
    print("\n" + "=" * 70)
    print("Per-query informativeness across all clips:")
    print("=" * 70)
    for q in QUERIES:
        answers = [r["answers"][q] for r in results]
        nonempty = sum(1 for a in answers if a and a.lower() not in ("none", ""))
        unique = len(set(answers))
        avg_len = sum(len(a) for a in answers) / max(len(answers), 1)
        print(f"  Q: {q[:55]:<58}")
        print(f"     non-empty {nonempty}/{len(answers)}, unique {unique}, avg len {avg_len:.0f} chars")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
