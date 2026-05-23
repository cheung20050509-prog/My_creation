#!/usr/bin/env python3
"""Generate per-token media and confidence assets for MOSEI sample 233939_0.

The script targets the qualitative case used in the PRISM/InfoGate analysis:
MOSEI test global_index=750, official utterance id 233939_0.  The processed pickle provides token/feature positions but no word-level
timestamps.  By default this script uses Whisper word timestamps (conda env
``qe_whisper``) to align each content token before cutting frames/audio.
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


def _modality_conf_payload(sens: dict[str, str], ib: dict[str, str]) -> dict[str, Any]:
    """Per-modality VTB confidence (``ib_conf_*``) plus InfoGate slot conf from sensitivity CSV."""
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


def _float(value: Any) -> float:
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def _maybe_number(value: str) -> Any:
    try:
        if value.strip() == "":
            return value
        return float(value)
    except (TypeError, ValueError):
        return value


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
    """Evenly spaced capture times inside (start, end), excluding exact endpoints."""
    if count <= 0:
        return []
    if count == 1:
        return [(start + end) / 2.0]
    span = end - start
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

    if sample_width == 1:
        dtype = np.uint8
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) - 128.0
        denom = 128.0
    elif sample_width == 2:
        dtype = np.int16
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        denom = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
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


def _load_sample(dataset_path: Path, global_index: int) -> tuple[list[str], float, list[str]]:
    with dataset_path.open("rb") as handle:
        data = pickle.load(handle)
    sample = data["test"][global_index]
    (words, _visual, _acoustic), label, segment = sample
    segment_list = [str(x) for x in np.asarray(segment).reshape(-1).tolist()]
    return list(words), _float(label), segment_list


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def generate(args: argparse.Namespace) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe must be available on PATH")

    dataset_path = Path(args.dataset_path)
    sensitivity_path = Path(args.sensitivity_csv)
    ib_path = Path(args.ib_tsv)
    video_path = Path(args.video)
    audio_path = Path(args.audio)
    output_root = Path(args.output_root)

    for path in (dataset_path, sensitivity_path, ib_path, video_path, audio_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    words, label, segment = _load_sample(dataset_path, args.global_index)
    sensitivity_rows = {int(row["pos"]): row for row in _read_csv(sensitivity_path)}
    ib_rows = {int(row["pos"]): row for row in _read_tsv(ib_path)}

    content_positions = list(range(1, len(words) + 1))
    if not all(pos in sensitivity_rows and pos in ib_rows for pos in content_positions):
        raise SystemExit("confidence rows do not cover all content token positions")

    video_duration = _probe_duration(video_path)
    audio_duration = _probe_duration(audio_path)
    media_duration = min(video_duration, audio_duration)
    token_duration = media_duration / float(len(words))

    output_root.mkdir(parents=True, exist_ok=True)

    alignment_mode = args.alignment
    whisper_timings: list[dict[str, Any]] | None = None
    if alignment_mode == "whisper":
        from whisper_token_align import run_alignment

        cache_path = Path(args.whisper_cache) if args.whisper_cache else None
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
                "timings": whisper_timings,
            },
        )

    # Derive original prediction from any intervention row:
    first_sens = sensitivity_rows[content_positions[0]]
    original_pred = float(first_sens["pred_masked"]) - float(first_sens["delta_pred"])
    utterance_id = f"{segment[0]}_0" if segment else "233939_0"

    manifest_rows: list[dict[str, Any]] = []
    manifest_json: dict[str, Any] = {
        "sample": {
            "dataset": "mosei",
            "split": "test",
            "global_index": args.global_index,
            "utterance_id": utterance_id,
            "video_id": segment[0] if segment else "233939",
            "segment": "0",
            "segment_start": _maybe_number(segment[1]) if len(segment) > 1 else None,
            "segment_end": _maybe_number(segment[2]) if len(segment) > 2 else None,
            "label": label,
            "prediction": original_pred,
            "media_duration": media_duration,
            "temporal_alignment": alignment_mode,
            "content_token_count": len(words),
            "source_video": str(video_path),
            "source_audio": str(audio_path),
            "frames_per_token": args.frames_per_token,
            "confidence_fields": "conf_t, conf_a, conf_v, conf_fused (VTB); conf_language/acoustic/visual (InfoGate slots)",
        },
        "tokens": [],
    }

    all_positions = [0] + content_positions + [len(words) + 1]
    for pos in all_positions:
        sens = sensitivity_rows[pos]
        ib = ib_rows[pos]
        is_content = pos in content_positions
        pkl_index = pos - 1 if is_content else None
        token = sens["token"]
        token_word = words[pkl_index] if pkl_index is not None else token

        folder_name = ""
        start = end = mid = None
        frame_times: list[float] = []
        frame_paths: list[str] = []
        if is_content:
            if whisper_timings is not None:
                timing = whisper_timings[pkl_index]
                start = float(timing["start"])
                end = float(timing["end"])
                mid = float(timing["mid"])
            else:
                start = pkl_index * token_duration
                end = (pkl_index + 1) * token_duration
                mid = (start + end) / 2.0
            frame_times = _frame_times(start, end, args.frames_per_token)
            folder_name = f"pos_{pos:02d}_{_safe_token_name(token_word)}"
            folder = output_root / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            frames_dir = folder / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            token_audio_path = folder / "audio.wav"
            waveform_path = folder / "waveform.png"

            for idx, t in enumerate(frame_times, start=1):
                frame_path = frames_dir / f"frame_{idx:02d}.jpg"
                _run([
                    ffmpeg,
                    "-y",
                    "-ss",
                    f"{t:.6f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ])
                frame_paths.append(str(frame_path.relative_to(folder)))

            # Midpoint preview at token root for quick browsing.
            _run([
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
            ])
            _run([
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
            ])
            _plot_waveform(token_audio_path, waveform_path, f"{pos:02d}: {token_word}")

        conf_payload = _modality_conf_payload(sens, ib)
        metadata_payload = {
            "dataset": "mosei",
            "split": "test",
            "global_index": args.global_index,
            "utterance_id": utterance_id,
            "video_id": segment[0] if segment else "233939",
            "label": label,
            "prediction": original_pred,
            "model_pos": pos,
            "pkl_token_index": pkl_index,
            "token": token,
            "token_word": token_word,
            "has_media": is_content,
            "time_start": start,
            "time_end": end,
            "time_mid": mid,
            "frame_count": len(frame_paths) if is_content else 0,
            "frame_times": frame_times if is_content else [],
            "frames": frame_paths if is_content else [],
            "time_alignment": alignment_mode if is_content else None,
            "whisper_word": whisper_timings[pkl_index]["whisper_word"]
            if is_content and whisper_timings is not None
            else None,
            "whisper_match": whisper_timings[pkl_index]["match"]
            if is_content and whisper_timings is not None
            else None,
            "source_video": str(video_path) if is_content else None,
            "source_audio": str(audio_path) if is_content else None,
            **conf_payload,
        }

        if is_content:
            folder = output_root / folder_name
            _write_json(folder / "metadata.json", metadata_payload)
            _write_json(folder / "conf.json", conf_payload)

        manifest_row = {
            "folder": folder_name,
            "has_media": int(is_content),
            "model_pos": pos,
            "pkl_token_index": "" if pkl_index is None else pkl_index,
            "token": token,
            "token_word": token_word,
            "time_start": "" if start is None else f"{start:.6f}",
            "time_end": "" if end is None else f"{end:.6f}",
            "time_mid": "" if mid is None else f"{mid:.6f}",
            "frame_count": len(frame_paths) if is_content else 0,
            **{k: conf_payload[k] for k in (
                "conf_t", "conf_a", "conf_v", "conf_fused",
                "conf_language", "conf_acoustic", "conf_visual", "aux_conf_mean",
            )},
        }
        manifest_rows.append(manifest_row)
        manifest_json["tokens"].append(
            {**metadata_payload, "folder": folder_name, "confidence": conf_payload}
        )

    fieldnames = list(manifest_rows[0].keys())
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    _write_json(output_root / "manifest.json", manifest_json)

    print(f"Wrote {len(words)} content token folders to {output_root}")
    print(f"Wrote {output_root / 'manifest.csv'}")
    print(f"Wrote {output_root / 'manifest.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-index", type=int, default=750)
    parser.add_argument("--dataset-path", default=str(MY_ROOT / "datasets" / "mosei.pkl"))
    parser.add_argument(
        "--sensitivity-csv",
        default=str(SCRIPT_DIR / "results" / "mosei_trial70_idx750_conf_token_sensitivity.csv"),
    )
    parser.add_argument(
        "--ib-tsv",
        default=str(SCRIPT_DIR / "results" / "mosei_trial70_idx750_ib_conf_tokens.tsv"),
    )
    parser.add_argument("--video", default=str(SCRIPT_DIR / "raw" / "233939" / "233939_0.mp4"))
    parser.add_argument("--audio", default=str(SCRIPT_DIR / "raw" / "233939" / "233939_0.wav"))
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "233939tokens"))
    parser.add_argument(
        "--frames-per-token",
        type=int,
        default=4,
        help="Number of video frames captured uniformly inside each token interval.",
    )
    parser.add_argument(
        "--alignment",
        choices=("whisper", "uniform"),
        default="whisper",
        help="Temporal alignment for frame/audio cuts (default: whisper word timestamps).",
    )
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument(
        "--whisper-cache",
        default=str(SCRIPT_DIR / "233939tokens" / "whisper_cache.json"),
    )
    args = parser.parse_args()
    if args.frames_per_token < 1:
        raise SystemExit("--frames-per-token must be >= 1")
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
