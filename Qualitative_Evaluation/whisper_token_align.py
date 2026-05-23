"""Whisper word-level alignment for MOSEI qualitative token folders."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _norm_token(word: str) -> str:
    w = word.lower().strip()
    w = re.sub(r"^▁|Ġ", "", w)
    w = re.sub(r"[^a-z0-9']+", "", w)
    replacements = {
        "im": "am",
        "its": "it",
        "reviewing": "review",
    }
    return replacements.get(w, w)


def _is_punct_only(word: str) -> bool:
    core = re.sub(r"[^a-zA-Z0-9]+", "", word)
    return core == ""


def extract_whisper_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words") or []:
            raw = str(w.get("word", "")).strip()
            if not raw:
                continue
            words.append(
                {
                    "word": raw,
                    "norm": _norm_token(raw),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                }
            )
    return words


def transcribe_with_whisper(
    audio_path: Path,
    *,
    model_name: str = "base",
    language: str = "en",
    device: str | None = None,
) -> dict[str, Any]:
    import whisper

    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=language,
        verbose=False,
    )
    return result


def align_pkl_words(
    pkl_words: list[str],
    whisper_words: list[dict[str, Any]],
    *,
    media_duration: float,
) -> list[dict[str, Any]]:
    """Map each pickle content word to [start, end] using Whisper + interpolation."""
    pkl_norm = [_norm_token(w) for w in pkl_words]
    w_norm = [w["norm"] for w in whisper_words]

    matcher = SequenceMatcher(a=pkl_norm, b=w_norm, autojunk=False)
    aligned: list[dict[str, Any] | None] = [None] * len(pkl_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            span = i2 - i1
            for offset in range(span):
                wi = whisper_words[j1 + offset]
                aligned[i1 + offset] = {
                    "start": wi["start"],
                    "end": wi["end"],
                    "whisper_word": wi["word"],
                    "match": "equal",
                }
        elif tag in ("replace", "delete"):
            w_span = j2 - j1
            p_span = i2 - i1
            if w_span == 0:
                continue
            if p_span == 0:
                continue
            for offset in range(p_span):
                pkl_i = i1 + offset
                if w_span == 1:
                    wi = whisper_words[j1]
                    start, end = wi["start"], wi["end"]
                    if p_span > 1:
                        dur = (end - start) / float(p_span)
                        start = start + offset * dur
                        end = start + dur
                    aligned[pkl_i] = {
                        "start": start,
                        "end": end,
                        "whisper_word": wi["word"],
                        "match": tag,
                    }
                else:
                    j = j1 + min(offset, w_span - 1)
                    wi = whisper_words[j]
                    aligned[pkl_i] = {
                        "start": wi["start"],
                        "end": wi["end"],
                        "whisper_word": wi["word"],
                        "match": tag,
                    }

    # Interpolate gaps (punctuation / unmatched tokens).
    for i, row in enumerate(aligned):
        if row is not None:
            continue
        prev_end = aligned[i - 1]["end"] if i > 0 and aligned[i - 1] else 0.0
        next_start = None
        for j in range(i + 1, len(aligned)):
            if aligned[j] is not None:
                next_start = aligned[j]["start"]
                break
        if next_start is None:
            next_start = media_duration
        if next_start <= prev_end:
            next_start = min(media_duration, prev_end + 0.05)
        aligned[i] = {
            "start": prev_end,
            "end": next_start,
            "whisper_word": None,
            "match": "interpolated",
        }

    out: list[dict[str, Any]] = []
    for idx, word in enumerate(pkl_words):
        row = aligned[idx]
        assert row is not None
        start = max(0.0, float(row["start"]))
        end = min(media_duration, max(float(row["end"]), start + 0.02))
        out.append(
            {
                "pkl_index": idx,
                "token_word": word,
                "start": start,
                "end": end,
                "mid": (start + end) / 2.0,
                "whisper_word": row.get("whisper_word"),
                "match": row.get("match"),
            }
        )
    return out


def run_alignment(
    audio_path: Path,
    pkl_words: list[str],
    *,
    model_name: str = "base",
    cache_path: Path | None = None,
    media_duration: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if cache_path and cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        transcript = cached["transcript"]
        timings = cached["timings"]
        return transcript, timings

    transcript = transcribe_with_whisper(audio_path, model_name=model_name)
    whisper_words = extract_whisper_words(transcript)
    if media_duration is None:
        media_duration = whisper_words[-1]["end"] if whisper_words else 0.0
    timings = align_pkl_words(pkl_words, whisper_words, media_duration=media_duration)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "transcript": transcript,
                    "whisper_words": whisper_words,
                    "timings": timings,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
            handle.write("\n")

    return transcript, timings
