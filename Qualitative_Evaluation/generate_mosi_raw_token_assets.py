#!/usr/bin/env python3
"""Generate per-token media for MOSI clips under Qualitative_Evaluation/raw/.

Uses Whisper word timestamps (conda env ``qe_whisper``) to align pickle word lists
before cutting frames and audio.  Confidence CSV/TSV inputs are optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MY_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _float(value: Any) -> float:
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def _probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(path),
        ],
        text=True,
    )
    return float(out.strip())


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _frame_times(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [(start + end) / 2.0]
    span = max(0.02, end - start)
    return [start + (i + 0.5) * span / float(count) for i in range(count)]


def _safe_token_name(token: str) -> str:
    clean = token.replace("▁", "").replace("Ġ", "").strip()
    replacements = {
        "": "blank",
        ".": "dot",
        ",": "comma",
        "?": "question",
        "!": "bang",
        "'": "quote",
        '"': "quote",
        "[CLS]": "cls",
        "[SEP]": "sep",
    }
    clean = replacements.get(clean, clean)
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", clean).strip("_").lower()
    return clean or "token"


def _plot_waveform(audio_path: Path, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate()
        n_channels = handle.getnchannels()
        n_frames = handle.getnframes()
        sample_width = handle.getsampwidth()
        raw = handle.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        denom = 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        denom = 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    audio = audio / denom
    times = np.arange(audio.size, dtype=np.float32) / float(sample_rate)

    fig, ax = plt.subplots(figsize=(5.2, 1.9), dpi=180)
    ax.plot(times, audio, linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.35)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time within token slice (s)", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_mosi_sample(dataset_path: Path, global_index: int) -> dict[str, Any]:
    with dataset_path.open("rb") as handle:
        data = pickle.load(handle)
    (words, _visual, _acoustic), label, segment = data["test"][global_index]
    segment_id = str(np.asarray(segment).reshape(-1)[0])
    return {
        "words": list(words),
        "label": _float(label),
        "segment_id": segment_id,
        "global_index": global_index,
    }


def _modality_conf_payload(sens: dict[str, str], ib: dict[str, str]) -> dict[str, Any]:
    return {
        "primary_name": sens["primary_name"],
        "conf_t": float(ib["ib_conf_t"]),
        "conf_a": float(ib["ib_conf_a"]),
        "conf_v": float(ib["ib_conf_v"]),
        "conf_fused": float(ib["ib_conf_fused"]),
        "conf_language": float(sens["conf_language"]),
        "conf_acoustic": float(sens["conf_acoustic"]),
        "conf_visual": float(sens["conf_visual"]),
        "aux_conf_mean": float(sens["aux_conf_mean"]),
    }


def _parse_raw_txt_line(path: Path) -> dict[str, str]:
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    clip_id, text, label, split = line.split(",", 3)
    return {
        "clip_id": clip_id.strip(),
        "raw_text": text.strip(),
        "label": label.strip(),
        "split": split.strip(),
    }


def generate(args: argparse.Namespace) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe must be available on PATH")

    video_path = Path(args.video)
    audio_path = Path(args.audio) if args.audio else video_path.with_suffix(".wav")
    output_root = Path(args.output_root)
    if not video_path.exists():
        raise SystemExit(f"missing video: {video_path}")
    if not audio_path.exists():
        raise SystemExit(f"missing audio: {audio_path}")

    if args.global_index is not None:
        sample = _load_mosi_sample(Path(args.dataset_path), args.global_index)
        words = sample["words"]
        label = sample["label"]
        segment_id = sample["segment_id"]
        global_index = sample["global_index"]
    else:
        meta = _parse_raw_txt_line(Path(args.raw_txt))
        words = meta["raw_text"].split()
        label = float(meta["label"])
        segment_id = meta["clip_id"]
        global_index = None

    sensitivity_rows: dict[int, dict[str, str]] = {}
    ib_rows: dict[int, dict[str, str]] = {}
    if args.sensitivity_csv and args.ib_tsv:
        sensitivity_rows = {int(row["pos"]): row for row in _read_csv(Path(args.sensitivity_csv))}
        ib_rows = {int(row["pos"]): row for row in _read_tsv(Path(args.ib_tsv))}

    video_duration = _probe_duration(video_path)
    audio_duration = _probe_duration(audio_path)
    media_duration = min(video_duration, audio_duration)
    output_root.mkdir(parents=True, exist_ok=True)

    from whisper_token_align import run_alignment

    cache_path = Path(args.whisper_cache) if args.whisper_cache else output_root / "whisper_cache.json"
    _transcript, whisper_timings = run_alignment(
        audio_path,
        words,
        model_name=args.whisper_model,
        cache_path=cache_path,
        media_duration=media_duration,
    )
    _write_json(
        output_root / "whisper_alignment.json",
        {
            "alignment": "whisper",
            "model": args.whisper_model,
            "audio": str(audio_path),
            "words": words,
            "timings": whisper_timings,
        },
    )

    content_positions = list(range(1, len(words) + 1))
    has_conf = bool(sensitivity_rows and ib_rows)
    if has_conf and not all(pos in sensitivity_rows and pos in ib_rows for pos in content_positions):
        raise SystemExit("confidence rows do not cover all content token positions")

    original_pred = None
    if has_conf:
        first_sens = sensitivity_rows[content_positions[0]]
        original_pred = float(first_sens["pred_masked"]) - float(first_sens["delta_pred"])

    manifest_json: dict[str, Any] = {
        "sample": {
            "dataset": "mosi",
            "split": "test",
            "global_index": global_index,
            "segment_id": segment_id,
            "clip_id": segment_id,
            "label": label,
            "prediction": original_pred,
            "raw_text": " ".join(words),
            "media_duration": media_duration,
            "temporal_alignment": "whisper",
            "content_token_count": len(words),
            "source_video": str(video_path),
            "source_audio": str(audio_path),
            "frames_per_token": args.frames_per_token,
        },
        "tokens": [],
    }
    manifest_rows: list[dict[str, Any]] = []

    for pos in content_positions:
        pkl_index = pos - 1
        token_word = words[pkl_index]
        timing = whisper_timings[pkl_index]
        start = float(timing["start"])
        end = float(timing["end"])
        mid = float(timing["mid"])
        frame_times = _frame_times(start, end, args.frames_per_token)
        folder_name = f"pos_{pos:02d}_{_safe_token_name(token_word)}"
        folder = output_root / folder_name
        frames_dir = folder / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, frame_time in enumerate(frame_times, start=1):
            frame_path = frames_dir / f"frame_{frame_idx:02d}.jpg"
            _run(
                [
                    ffmpeg,
                    "-y",
                    "-ss",
                    f"{frame_time:.6f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ]
            )

        _run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{mid:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(folder / "frame.jpg"),
            ]
        )
        token_audio_path = folder / "audio.wav"
        _run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{start:.6f}",
                "-t",
                f"{end - start:.6f}",
                "-i",
                str(audio_path),
                "-acodec",
                "pcm_s16le",
                str(token_audio_path),
            ]
        )
        _plot_waveform(token_audio_path, folder / "waveform.png", f"{pos:02d}: {token_word}")

        conf_payload: dict[str, Any] = {}
        token = token_word
        if has_conf:
            sens = sensitivity_rows[pos]
            ib = ib_rows[pos]
            token = ib["token"]
            conf_payload = _modality_conf_payload(sens, ib)

        metadata_payload = {
            "dataset": "mosi",
            "split": "test",
            "global_index": global_index,
            "segment_id": segment_id,
            "label": label,
            "prediction": original_pred,
            "model_pos": pos,
            "pkl_token_index": pkl_index,
            "token": token,
            "token_word": token_word,
            "has_media": True,
            "time_start": start,
            "time_end": end,
            "time_mid": mid,
            "time_alignment": "whisper",
            "whisper_word": timing.get("whisper_word"),
            "whisper_match": timing.get("match"),
            "source_video": str(video_path),
            "source_audio": str(audio_path),
            **conf_payload,
        }
        _write_json(folder / "metadata.json", metadata_payload)
        if conf_payload:
            _write_json(folder / "conf.json", conf_payload)

        manifest_row = {
            "folder": folder_name,
            "model_pos": pos,
            "pkl_token_index": pkl_index,
            "token": token,
            "token_word": token_word,
            "time_start": f"{start:.6f}",
            "time_end": f"{end:.6f}",
            "time_mid": f"{mid:.6f}",
            "whisper_word": timing.get("whisper_word") or "",
            "whisper_match": timing.get("match") or "",
            **conf_payload,
        }
        manifest_rows.append(manifest_row)
        manifest_json["tokens"].append({**metadata_payload, "folder": folder_name})

    fieldnames = list(manifest_rows[0].keys())
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    _write_json(output_root / "manifest.json", manifest_json)

    print(f"Wrote {len(words)} token folders to {output_root}")
    print(f"segment_id={segment_id} global_index={global_index}")


def main() -> int:
    raw_dir = SCRIPT_DIR / "raw" / "d3_k5Xpfmik"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-index", type=int, default=116)
    parser.add_argument("--dataset-path", default=str(MY_ROOT / "datasets" / "mosi.pkl"))
    parser.add_argument("--raw-txt", default=str(raw_dir / "d3_k5Xpfmik.txt"))
    parser.add_argument("--video", default=str(raw_dir / "d3_k5Xpfmik_15.mp4"))
    parser.add_argument("--audio", default=str(raw_dir / "d3_k5Xpfmik_15.wav"))
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "d3_k5Xpfmik_15tokens"))
    parser.add_argument("--sensitivity-csv", default=None)
    parser.add_argument("--ib-tsv", default=None)
    parser.add_argument("--frames-per-token", type=int, default=4)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-cache", default=None)
    args = parser.parse_args()
    if args.frames_per_token < 1:
        raise SystemExit("--frames-per-token must be >= 1")
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
