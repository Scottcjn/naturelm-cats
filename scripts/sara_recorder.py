"""Sara recorder — interactive corpus-collection tool for cat vocalizations.

Captures audio + your contextual notes per clip. Saves to ~/sara-corpus/{timestamp}/
with audio.wav (16 kHz mono) and meta.json.

Designed for the real research question: does Sara have a personal vocalization
repertoire with distinguishable categories matching observed contexts? And do any
of her vocalizations show TURN-TAKING patterns (like Maddie's eeh-ooo → ack → grint
grint)?

Usage:
    cd /home/scott/naturelm-cats
    uv run python scripts/sara_recorder.py

    # List input devices first if you want to pick one:
    uv run python scripts/sara_recorder.py --list-devices

    # Use a specific input device:
    uv run python scripts/sara_recorder.py --device 5

Flow:
    1. Press ENTER  → start recording (16 kHz mono)
    2. Press ENTER  → stop recording (max 60s safety cap)
    3. Answer prompts: what was happening / your interpretation / turn-taking?
    4. Saved to ~/sara-corpus/{ISO-timestamp}/

Append-only corpus. Run as many times as you want.
"""
import argparse
import datetime as dt
import json
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

CORPUS_DIR = Path.home() / "sara-corpus"
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 60  # safety cap


def list_devices() -> None:
    print(sd.query_devices())


def record_until_enter(samplerate: int, device: int | None) -> tuple[np.ndarray, float]:
    """Record from the mic until user presses ENTER. Returns (audio, duration_seconds)."""
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, n_frames, time_info, status):
        if status:
            print(f"[status] {status}", file=sys.stderr)
        frames.append(indata.copy())

    print("\n  Recording... press ENTER to stop (or wait, max 60s).", flush=True)

    # Run InputStream in context manager; main thread waits for ENTER (or timeout)
    def waiter():
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        stop_event.set()

    waiter_thread = threading.Thread(target=waiter, daemon=True)
    waiter_thread.start()

    with sd.InputStream(samplerate=samplerate, channels=1, device=device,
                        callback=callback, dtype="float32"):
        elapsed = 0.0
        while not stop_event.is_set() and elapsed < MAX_RECORD_SECONDS:
            sd.sleep(100)
            elapsed = sum(f.shape[0] for f in frames) / samplerate
            print(f"\r  ⏺  {elapsed:5.1f}s  ", end="", flush=True)

    print()  # newline after the ⏺ progress line
    if not frames:
        return np.zeros(0, dtype=np.float32), 0.0
    audio = np.concatenate(frames, axis=0).squeeze()
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, len(audio) / samplerate


def play_preview(audio: np.ndarray, samplerate: int) -> None:
    print("  Playing preview... (Ctrl+C to skip)")
    try:
        sd.play(audio, samplerate=samplerate)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("  (skipped)")


def prompt(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {question}{suffix}: ").strip()
    return val or default


def prompt_yes_no(question: str, default: bool = False) -> bool:
    d = "y/N" if not default else "Y/n"
    val = input(f"  {question} ({d}): ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


def prompt_float(question: str) -> float | None:
    val = input(f"  {question}: ").strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        print("  (not a number, skipping)")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    ap.add_argument("--device", type=int, default=None,
                    help="Input device index (use --list-devices to find one). Default: system default.")
    ap.add_argument("--cat", default="Sara",
                    help="Which cat is this recording for? (default: Sara)")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Sara recorder  —  cat: {args.cat}")
    print(f"  Corpus dir: {CORPUS_DIR}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz  ·  Mono")
    if args.device is not None:
        print(f"  Input device: {args.device}")
    print("=" * 70)

    while True:
        print()
        ready = input("  Press ENTER to start a new recording (or 'q' to quit): ").strip().lower()
        if ready == "q":
            print("  Goodbye.")
            return 0

        audio, duration = record_until_enter(SAMPLE_RATE, args.device)
        if duration < 0.3:
            print("  Recording too short — discarded.")
            continue

        print(f"  Captured {duration:.2f}s @ {SAMPLE_RATE} Hz")
        # Quick stats
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        print(f"  Peak: {peak:.3f}  RMS: {rms:.4f}")

        if peak < 0.001:
            print("  WARNING: signal extremely quiet — check microphone.")

        if prompt_yes_no("Play it back to confirm?", default=False):
            play_preview(audio, SAMPLE_RATE)

        if not prompt_yes_no("Save this recording?", default=True):
            print("  Discarded.")
            continue

        # Metadata prompts
        print("\n  --- Context ---")
        what = prompt("What was happening? (free text)")
        interp = prompt("Your interpretation? (e.g. greeting, food-demand, play, alarm)")
        is_exchange = prompt_yes_no("Was this part of a turn-taking exchange (cat → ack → cat-again)?")

        ack_time = None
        followup_form = None
        if is_exchange:
            ack_time = prompt_float("Approx. seconds into the clip when YOU acknowledged her")
            followup_form = prompt("Did she vocalize again AFTER your ack? Describe form (e.g. 'eeh-ooo', 'grint grint', 'longer mrrow')")

        compare_maddie = prompt("Does this remind you of any Maddie pattern? (free text, or leave blank)")
        notes = prompt("Anything else? (free text, or leave blank)")

        # Save
        timestamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        clip_dir = CORPUS_DIR / timestamp
        clip_dir.mkdir(parents=True, exist_ok=True)

        sf.write(str(clip_dir / "audio.wav"), audio, SAMPLE_RATE, subtype="PCM_16")

        meta = {
            "timestamp": dt.datetime.now().isoformat(),
            "cat": args.cat,
            "duration_seconds": round(duration, 3),
            "sample_rate": SAMPLE_RATE,
            "peak_amplitude": round(peak, 4),
            "rms": round(rms, 5),
            "context_what": what,
            "interpretation": interp,
            "is_turn_taking_exchange": is_exchange,
            "ack_seconds_into_clip": ack_time,
            "followup_vocalization_form": followup_form,
            "compares_to_maddie": compare_maddie,
            "notes": notes,
            "input_device": args.device,
            "tool_version": "sara_recorder.py v0.1",
        }
        (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"  ✓ Saved to {clip_dir}/")
        print(f"     audio.wav  ({duration:.2f}s)")
        print(f"     meta.json")


if __name__ == "__main__":
    raise SystemExit(main())
