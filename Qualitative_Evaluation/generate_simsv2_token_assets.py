#!/usr/bin/env python3
"""Generate per-token media and confidence assets for SIMSv2 sample 673."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import shutil
import subprocess
import sys
import unicodedata
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MY_ROOT = SCRIPT_DIR.parent


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj),
        )
        handle.write("\n")


def _float(value: Any) -> float:
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0])


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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


def _frame_times(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [(start + end) / 2.0]
    span = max(0.02, end - start)
    return [start + (i + 0.5) * span / float(count) for i in range(count)]


def _is_punct(token: str) -> bool:
    return bool(token) and all(unicodedata.category(ch).startswith("P") for ch in token)


def _safe_token_name(token: str) -> str:
    replacements = {
        "": "blank",
        ".": "dot",
        "。": "dot",
        ",": "comma",
        "，": "comma",
        "?": "question",
        "？": "question",
        "!": "bang",
        "！": "bang",
        "[CLS]": "cls",
        "[SEP]": "sep",
    }
    if token in replacements:
        return replacements[token]
    clean = token.replace("▁", "").replace("Ġ", "").strip()
    ascii_clean = re.sub(r"[^A-Za-z0-9_-]+", "_", clean).strip("_").lower()
    if ascii_clean:
        return ascii_clean
    if len(clean) == 1:
        return f"u{ord(clean):04x}"
    return "_".join(f"u{ord(ch):04x}" for ch in clean) or "token"


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
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        denom = 128.0
    elif sample_width == 2:
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


def _load_simsv2_sample(dataset_path: Path, split: str, global_index: int) -> dict[str, Any]:
    with dataset_path.open("rb") as handle:
        data = pickle.load(handle)
    split_data = data[split]
    return {
        "id": str(split_data["id"][global_index]),
        "raw_text": str(split_data["raw_text"][global_index]),
        "label": _float(split_data["regression_labels"][global_index]),
        "annotation": str(split_data["annotations"][global_index]),
    }


def _token_rows(ib_path: Path, sensitivity_path: Path) -> tuple[list[dict[str, str]], dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    ib_rows_all = _read_tsv(ib_path)
    sensitivity_rows = {int(row["pos"]): row for row in _read_csv(sensitivity_path)}
    ib_rows = {int(row["pos"]): row for row in ib_rows_all}
    valid_rows = [
        row
        for row in ib_rows_all
        if int(row.get("valid", "0")) == 1 and row["token"] != "[PAD]"
    ]
    return valid_rows, sensitivity_rows, ib_rows


def _extract_audio(ffmpeg: str, video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(audio_path),
        ]
    )


def _funasr_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _run_funasr(args: argparse.Namespace, audio_path: Path, cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    from funasr import AutoModel

    device = _funasr_device(args.device)
    model_kwargs: dict[str, Any] = {
        "model": args.funasr_model,
        "vad_model": args.funasr_vad_model,
        "punc_model": args.funasr_punc_model,
        "device": device,
    }
    if args.funasr_timestamp_model:
        model_kwargs["timestamp_model"] = args.funasr_timestamp_model

    model = AutoModel(**model_kwargs)
    result = model.generate(input=str(audio_path), batch_size_s=args.batch_size_s)
    payload = {
        "device": device,
        "model_kwargs": model_kwargs,
        "audio": str(audio_path),
        "result": result,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cache_path, payload)
    return payload


def _flatten_funasr_result(payload: dict[str, Any], media_duration: float) -> list[dict[str, Any]]:
    result = payload.get("result", [])
    if isinstance(result, dict):
        result = [result]

    text_parts: list[str] = []
    timestamps: list[Any] = []
    for row in result:
        if not isinstance(row, dict):
            continue
        text_parts.append(str(row.get("text", "")))
        ts = row.get("timestamp") or row.get("timestamps") or []
        if isinstance(ts, list):
            timestamps.extend(ts)

    chars = [ch for ch in "".join(text_parts) if not ch.isspace()]
    non_punct_indices = [idx for idx, ch in enumerate(chars) if not _is_punct(ch)]
    timestamp_scale = 1.0
    flat_ts = [ts for ts in timestamps if isinstance(ts, (list, tuple)) and len(ts) >= 2]
    if flat_ts:
        max_end = max(float(ts[1]) for ts in flat_ts)
        if max_end > media_duration + 5.0:
            timestamp_scale = 1000.0

    items = [{"char": ch, "start": None, "end": None, "match": "funasr_text"} for ch in chars]
    if len(flat_ts) == len(chars):
        indices = list(range(len(chars)))
    elif len(flat_ts) == len(non_punct_indices):
        indices = non_punct_indices
    else:
        indices = list(range(min(len(flat_ts), len(chars))))

    for idx, ts in zip(indices, flat_ts):
        start = float(ts[0]) / timestamp_scale
        end = float(ts[1]) / timestamp_scale
        items[idx].update({"start": start, "end": end, "match": "funasr_timestamp"})
    return items


def _norm_char(ch: str) -> str:
    replacements = {
        ",": "，",
        ".": "。",
        "?": "？",
        "!": "！",
        ":": "：",
        ";": "；",
    }
    return replacements.get(ch.strip(), ch.strip())


def _strip_punct(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace() and not _is_punct(ch))


def _interpolate_missing(
    aligned: list[dict[str, Any] | None],
    tokens: list[str],
    media_duration: float,
) -> list[dict[str, Any]]:
    out = aligned[:]
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i + 1
        while j < len(out) and out[j] is None:
            j += 1
        prev_end = out[i - 1]["end"] if i > 0 and out[i - 1] is not None else 0.0
        next_start = out[j]["start"] if j < len(out) and out[j] is not None else media_duration
        if next_start <= prev_end:
            next_start = min(media_duration, prev_end + 0.05 * (j - i))
        step = max(0.02, (next_start - prev_end) / float(j - i))
        for k in range(i, j):
            start = min(media_duration, prev_end + (k - i) * step)
            end = min(media_duration, start + step)
            if end <= start:
                end = min(media_duration, start + 0.02)
            out[k] = {
                "start": start,
                "end": end,
                "whisper_word": None,
                "funasr_char": None,
                "match": "interpolated",
            }
        i = j

    final: list[dict[str, Any]] = []
    for idx, row in enumerate(out):
        assert row is not None
        start = max(0.0, float(row["start"]))
        end = min(media_duration, max(float(row["end"]), start + 0.02))
        final.append(
            {
                "pkl_index": idx,
                "token_word": tokens[idx],
                "start": start,
                "end": end,
                "mid": (start + end) / 2.0,
                "whisper_word": row.get("whisper_word"),
                "funasr_char": row.get("funasr_char"),
                "match": row.get("match"),
            }
        )
    return final


def align_tokens_with_funasr(
    tokens: list[str],
    funasr_items: list[dict[str, Any]],
    *,
    media_duration: float,
) -> list[dict[str, Any]]:
    target_norm = [_norm_char(token) for token in tokens]
    asr_norm = [_norm_char(str(item["char"])) for item in funasr_items]
    matcher = SequenceMatcher(a=target_norm, b=asr_norm, autojunk=False)
    aligned: list[dict[str, Any] | None] = [None] * len(tokens)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                item = funasr_items[j1 + offset]
                if item["start"] is None or item["end"] is None:
                    continue
                aligned[i1 + offset] = {
                    "start": item["start"],
                    "end": item["end"],
                    "whisper_word": item["char"],
                    "funasr_char": item["char"],
                    "match": item["match"],
                }
            continue

        if tag not in {"replace", "delete"}:
            continue
        span_items = funasr_items[j1:j2]
        used: set[int] = set()
        for target_idx in range(i1, i2):
            target = target_norm[target_idx]
            for local_idx, item in enumerate(span_items):
                if local_idx in used:
                    continue
                if _norm_char(str(item["char"])) != target:
                    continue
                if item["start"] is None or item["end"] is None:
                    continue
                aligned[target_idx] = {
                    "start": item["start"],
                    "end": item["end"],
                    "whisper_word": item["char"],
                    "funasr_char": item["char"],
                    "match": f"{tag}_char",
                }
                used.add(local_idx)
                break

        # For short Chinese ASR substitutions (e.g. 呢 -> 啦), keep the available
        # acoustic timestamp on the nearest unmatched non-punctuation token.
        remaining_items = [
            (local_idx, item)
            for local_idx, item in enumerate(span_items)
            if local_idx not in used and item["start"] is not None and item["end"] is not None
        ]
        remaining_targets = [
            target_idx
            for target_idx in range(i1, i2)
            if aligned[target_idx] is None and not _is_punct(tokens[target_idx])
        ]
        for target_idx, (local_idx, item) in zip(remaining_targets, remaining_items):
            aligned[target_idx] = {
                "start": item["start"],
                "end": item["end"],
                "whisper_word": item["char"],
                "funasr_char": item["char"],
                "match": f"{tag}_timed_substitution",
            }
            used.add(local_idx)

    return _interpolate_missing(aligned, tokens, media_duration)


def generate(args: argparse.Namespace) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe must be available on PATH")

    dataset_path = Path(args.dataset_path)
    sensitivity_path = Path(args.sensitivity_csv)
    ib_path = Path(args.ib_tsv)
    video_path = Path(args.video)
    output_root = Path(args.output_root)
    for path in (dataset_path, sensitivity_path, ib_path, video_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    output_root.mkdir(parents=True, exist_ok=True)
    source_audio = Path(args.source_audio) if args.source_audio else output_root / "source.wav"
    _extract_audio(ffmpeg, video_path, source_audio)

    sample = _load_simsv2_sample(dataset_path, args.split, args.global_index)
    valid_rows, sensitivity_rows, ib_rows = _token_rows(ib_path, sensitivity_path)
    positions = [int(row["pos"]) for row in valid_rows]
    all_positions = positions
    content_positions = [
        int(row["pos"])
        for row in valid_rows
        if row["token"] not in {"[CLS]", "[SEP]"}
    ]
    content_tokens = [ib_rows[pos]["token"] for pos in content_positions]

    if "".join(token for token in content_tokens if not _is_punct(token)) != _strip_punct(sample["raw_text"]):
        raise SystemExit("content tokens do not match SIMSv2 raw_text")

    video_duration = _probe_duration(video_path)
    audio_duration = _probe_duration(source_audio)
    media_duration = min(video_duration, audio_duration)
    cache_path = Path(args.funasr_cache) if args.funasr_cache else output_root / "funasr_cache.json"
    funasr_payload = _run_funasr(args, source_audio, cache_path)
    funasr_items = _flatten_funasr_result(funasr_payload, media_duration)
    timings = align_tokens_with_funasr(content_tokens, funasr_items, media_duration=media_duration)
    _write_json(
        output_root / "funasr_alignment.json",
        {
            "alignment": "funasr",
            "cache": str(cache_path),
            "content_tokens": content_tokens,
            "funasr_items": funasr_items,
            "timings": timings,
        },
    )

    first_sens = sensitivity_rows[all_positions[0]]
    original_pred = float(first_sens["pred_masked"]) - float(first_sens["delta_pred"])
    sample_id = sample["id"]
    video_id, clip_id = sample_id.split("$_$", 1) if "$_$" in sample_id else (sample_id, "")
    utterance_id = f"{video_id}_{clip_id}" if clip_id else sample_id

    manifest_rows: list[dict[str, Any]] = []
    manifest_json: dict[str, Any] = {
        "sample": {
            "dataset": "simsv2",
            "split": args.split,
            "global_index": args.global_index,
            "utterance_id": utterance_id,
            "video_id": video_id,
            "clip_id": clip_id,
            "sample_id": sample_id,
            "raw_text": sample["raw_text"],
            "annotation": sample["annotation"],
            "label": sample["label"],
            "prediction": original_pred,
            "media_duration": media_duration,
            "temporal_alignment": "funasr",
            "content_token_count": len(content_tokens),
            "source_video": str(video_path),
            "source_audio": str(source_audio),
            "frames_per_token": args.frames_per_token,
            "confidence_fields": "conf_t, conf_a, conf_v, conf_fused (VTB); conf_language/acoustic/visual (InfoGate slots)",
        },
        "tokens": [],
    }

    timing_by_pos = {
        pos: timings[offset]
        for offset, pos in enumerate(content_positions)
    }

    for pos in all_positions:
        sens = sensitivity_rows[pos]
        ib = ib_rows[pos]
        token = ib["token"]
        is_content = pos in timing_by_pos
        pkl_index = content_positions.index(pos) if is_content else None
        token_word = content_tokens[pkl_index] if pkl_index is not None else token

        folder_name = ""
        start = end = mid = None
        frame_times: list[float] = []
        frame_paths: list[str] = []
        timing = timing_by_pos.get(pos)
        if timing is not None:
            start = float(timing["start"])
            end = float(timing["end"])
            mid = float(timing["mid"])
            frame_times = _frame_times(start, end, args.frames_per_token)
            folder_name = f"pos_{pos:02d}_{_safe_token_name(token_word)}"
            folder = output_root / folder_name
            frames_dir = folder / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            token_audio_path = folder / "audio.wav"
            waveform_path = folder / "waveform.png"

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
                frame_paths.append(str(frame_path.relative_to(folder)))

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
            _run(
                [
                    ffmpeg,
                    "-y",
                    "-ss",
                    f"{start:.6f}",
                    "-t",
                    f"{end - start:.6f}",
                    "-i",
                    str(source_audio),
                    "-acodec",
                    "pcm_s16le",
                    str(token_audio_path),
                ]
            )
            _plot_waveform(token_audio_path, waveform_path, f"{pos:02d}: {_safe_token_name(token_word)}")

        conf_payload = _modality_conf_payload(sens, ib)
        metadata_payload = {
            "dataset": "simsv2",
            "split": args.split,
            "global_index": args.global_index,
            "utterance_id": utterance_id,
            "video_id": video_id,
            "clip_id": clip_id,
            "sample_id": sample_id,
            "raw_text": sample["raw_text"],
            "annotation": sample["annotation"],
            "label": sample["label"],
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
            "time_alignment": "funasr" if is_content else None,
            "whisper_word": timing.get("whisper_word") if timing else None,
            "funasr_char": timing.get("funasr_char") if timing else None,
            "funasr_match": timing.get("match") if timing else None,
            "source_video": str(video_path) if is_content else None,
            "source_audio": str(source_audio) if is_content else None,
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
            **{
                k: conf_payload[k]
                for k in (
                    "conf_t",
                    "conf_a",
                    "conf_v",
                    "conf_fused",
                    "conf_language",
                    "conf_acoustic",
                    "conf_visual",
                    "aux_conf_mean",
                )
            },
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

    print(f"Wrote {len(content_tokens)} content token folders to {output_root}")
    print(f"Wrote {output_root / 'manifest.csv'}")
    print(f"Wrote {output_root / 'manifest.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-index", type=int, default=673)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-path", default=str(MY_ROOT / "datasets" / "simsv2.pkl"))
    parser.add_argument(
        "--sensitivity-csv",
        default=str(SCRIPT_DIR / "results" / "simsv2_trial53_idx673_conf_token_sensitivity.csv"),
    )
    parser.add_argument(
        "--ib-tsv",
        default=str(SCRIPT_DIR / "results" / "simsv2_trial53_idx673_ib_conf_tokens.tsv"),
    )
    parser.add_argument(
        "--video",
        default=str(
            SCRIPT_DIR
            / "raw"
            / "aqgy5_0008_00033"
            / "Ch-sims_aqgy5_0008_00033.mp4"
        ),
    )
    parser.add_argument("--source-audio", default=None)
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "aqgy5_0008_00033tokens"))
    parser.add_argument("--frames-per-token", type=int, default=4)
    parser.add_argument("--funasr-cache", default=None)
    parser.add_argument("--funasr-model", default="paraformer-zh")
    parser.add_argument("--funasr-vad-model", default="fsmn-vad")
    parser.add_argument("--funasr-punc-model", default="ct-punc")
    parser.add_argument("--funasr-timestamp-model", default="fa-zh")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size-s", type=int, default=60)
    args = parser.parse_args()
    if args.frames_per_token < 1:
        raise SystemExit("--frames-per-token must be >= 1")
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
