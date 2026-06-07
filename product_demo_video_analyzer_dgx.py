import argparse
import difflib
import html
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from youtube_transcript_helpers import fetch_transcript_entries

try:
    import cv2
except Exception:
    cv2 = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


def build_qwen_engine(
    model_id: str,
    quantization: str | None = None,
    max_model_len: int = 32768,
    max_images: int = 32,
    gpu_memory_utilization: float = 0.9,
):
    """Load Qwen2.5-VL via vLLM. On Blackwell, AWQ checkpoints auto-select the Marlin kernel;
    pass quantization='nvfp4' with a pre-quantized NVFP4 checkpoint to hit the FP4 tensor cores."""
    if LLM is None:
        raise RuntimeError("vllm not installed. Run: pip install 'vllm>=0.6.4'")

    print(
        f"[qwen] Loading '{model_id}' via vLLM "
        f"(quantization={quantization or 'auto'}, max_model_len={max_model_len})..."
    )
    return LLM(
        model=model_id,
        dtype="auto",
        quantization=quantization,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": max_images},
        trust_remote_code=True,
        allowed_local_media_path="/",
    )


def call_qwen(
    engine,
    prompt: str,
    image_paths: list[Path],
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> str:
    """Run vLLM chat inference on Qwen2.5-VL with N images + a text prompt."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"file://{Path(p).resolve()}"}}
        for p in image_paths
    ]
    content.append({"type": "text", "text": prompt})

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = engine.chat(
        messages=[{"role": "user", "content": content}],
        sampling_params=sampling_params,
    )
    return outputs[0].outputs[0].text.strip()


def build_ocr_engine():
    """Load RapidOCR (ONNX Runtime, CPU by default). ARM-friendly, wheel-only setup."""
    if RapidOCR is None:
        raise RuntimeError("rapidocr-onnxruntime not installed. Run: pip install rapidocr-onnxruntime")
    return RapidOCR()


def compute_dhash(frame_bgr, hash_size: int = 8) -> np.ndarray:
    """Difference hash — cheap perceptual fingerprint. Returns a flat bool array of length hash_size**2."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    return (resized[:, 1:] > resized[:, :-1]).flatten()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).sum())


def require_yt_dlp():
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: yt_dlp. Install with: pip install -U yt-dlp"
        ) from exc
    return yt_dlp


def require_transcript_api():
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: youtube_transcript_api. Install with: "
            "pip install -U youtube-transcript-api"
        ) from exc
    return YouTubeTranscriptApi


def extract_video_id(video_url: str) -> str:
    import hashlib
    parsed = urlparse(video_url)

    # YouTube: youtu.be/<id>
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.lstrip("/").split("/")[0]

    # YouTube: youtube.com/watch?v=<id>
    if "youtube.com" in parsed.netloc:
        query_id = parse_qs(parsed.query).get("v")
        if query_id and query_id[0]:
            return query_id[0]

    # Wistia embed: fast.wistia.com/embed/iframe/<id>
    if "wistia.com" in parsed.netloc:
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            return parts[-1]

    # Wistia embedded in a page: ?wvideo=<id>
    wvideo = parse_qs(parsed.query).get("wvideo")
    if wvideo and wvideo[0]:
        return wvideo[0]

    # Generic fallback: use last non-empty path segment, cleaned up
    path_parts = [p for p in parsed.path.split("/") if p]
    if path_parts:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", path_parts[-1])[:40]
        if slug:
            return slug

    # Last resort: short hash of the full URL
    return hashlib.md5(video_url.encode()).hexdigest()[:12]


def clean_transcript_text(text: str) -> str:
    cleaned = re.sub(r"\[[^\]]+\]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def seconds_to_hhmmss(seconds: float) -> str:
    total = int(max(0, seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_timecode_to_seconds(value: str) -> float:
    normalized = value.strip().replace(",", ".")
    parts = normalized.split(":")
    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    return float(normalized)


def normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    previous_text = None
    previous_start = None

    for segment in sorted(segments, key=lambda item: item["start"]):
        text = clean_transcript_text(segment.get("text", ""))
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start + 2.0))
        if end <= start:
            end = start + 2.0

        is_duplicate = (
            previous_text == text
            and previous_start is not None
            and abs(start - previous_start) < 0.3
        )
        if is_duplicate:
            continue

        normalized.append({"start": start, "end": end, "text": text})
        previous_text = text
        previous_start = start

    return normalized


def join_segments_text(segments: list[dict[str, Any]]) -> str:
    return clean_transcript_text(" ".join(item["text"] for item in segments))


def get_transcript_segments_from_api(video_id: str) -> Optional[list[dict[str, Any]]]:
    try:
        items = fetch_transcript_entries(video_id, languages=("en", "en-US"))
    except Exception:
        return None

    segments = []
    for item in items:
        start = float(item.get("start", 0.0))
        duration = float(item.get("duration", 2.0))
        segments.append(
            {
                "start": start,
                "end": start + max(0.2, duration),
                "text": item.get("text", ""),
            }
        )
    return normalize_segments(segments) or None


def parse_vtt_segments(vtt_text: str) -> list[dict[str, Any]]:
    time_pattern = re.compile(
        r"^\s*((?:\d{2}:)?\d{2}:\d{2}[.,]\d{3})\s*-->\s*((?:\d{2}:)?\d{2}:\d{2}[.,]\d{3})"
    )
    lines = vtt_text.splitlines()
    segments: list[dict[str, Any]] = []

    current_start = None
    current_end = None
    current_text_lines: list[str] = []

    def flush() -> None:
        nonlocal current_start, current_end, current_text_lines
        if current_start is None or current_end is None:
            current_text_lines = []
            return
        text_raw = " ".join(current_text_lines)
        text_raw = re.sub(r"<[^>]+>", "", text_raw)
        text_raw = html.unescape(text_raw)
        text = clean_transcript_text(text_raw)
        if text:
            segments.append({"start": current_start, "end": current_end, "text": text})
        current_start = None
        current_end = None
        current_text_lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        match = time_pattern.match(line)
        if match:
            flush()
            current_start = parse_timecode_to_seconds(match.group(1))
            current_end = parse_timecode_to_seconds(match.group(2))
            continue
        if line.isdigit():
            continue
        if current_start is not None:
            line = re.sub(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}>", "", line)
            current_text_lines.append(line)

    flush()
    return normalize_segments(segments)


def get_transcript_segments_from_ytdlp(video_url: str, work_dir: Path) -> Optional[list[dict[str, Any]]]:
    try:
        yt_dlp = require_yt_dlp()
    except RuntimeError as exc:
        print(exc)
        return None

    transcript_dir = work_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    video_id = extract_video_id(video_url)

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US"],
        "subtitlesformat": "vtt",
        "outtmpl": str(transcript_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_url])
    except Exception:
        return None

    candidates = sorted(transcript_dir.glob(f"{video_id}*.vtt"))
    for path in candidates:
        try:
            segments = parse_vtt_segments(path.read_text(encoding="utf-8", errors="ignore"))
            if segments:
                return segments
        except Exception:
            continue
    return None


def download_video(video_url: str, video_id: str, work_dir: Path, max_height: int = 720) -> Optional[Path]:
    try:
        yt_dlp = require_yt_dlp()
    except RuntimeError as exc:
        print(exc)
        return None

    video_dir = work_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_template = video_dir / f"{video_id}.%(ext)s"

    option_sets = [
        {
            # progressive mp4 fallback (often avoids merge dependency issues)
            "format": f"best[ext=mp4][height<={max_height}]/best[height<={max_height}]/best",
        },
        {
            # best quality under max_height, with merge if split streams
            "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
            "merge_output_format": "mp4",
        },
        {
            # last-resort generic best
            "format": "best",
        },
    ]

    last_error = None
    for idx, option_set in enumerate(option_sets, start=1):
        opts = {
            **option_set,
            "outtmpl": str(output_template),
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([video_url])
            candidates = sorted(video_dir.glob(f"{video_id}.*"))
            if candidates:
                return candidates[0]
        except Exception as exc:
            last_error = exc
            print(f"[{video_id}] Video download attempt {idx} failed: {exc}")

    if last_error is not None:
        print(
            f"[{video_id}] All video download strategies failed. "
            "If this is a restricted/private video, provide accessible URL/cookies."
        )
    return None


def _whisper_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def transcribe_with_whisper(video_path: Path, whisper_model: str) -> Optional[list[dict[str, Any]]]:
    try:
        import whisper
    except Exception:
        print(
            "Whisper fallback requested but the 'whisper' package is not installed. "
            "Install with: pip install -U openai-whisper"
        )
        return None

    try:
        device = _whisper_device()
        print(f"[whisper] Loading model '{whisper_model}' on device: {device}")
        model = whisper.load_model(whisper_model, device=device)
        result = model.transcribe(str(video_path), verbose=False, fp16=(device != "cpu"))
        raw_segments = result.get("segments", [])
        segments = []
        for item in raw_segments:
            segments.append(
                {
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "text": item.get("text", ""),
                }
            )
        return normalize_segments(segments) or None
    except Exception as exc:
        print(f"Whisper failed: {exc}")
        return None


def extract_audio_for_whisper(video_path: Path, audio_dir: Path, video_id: str) -> Optional[Path]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = audio_dir / f"{video_id}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None
    if not output_path.exists():
        return None
    return output_path


def transcribe_with_whisper_audio(
    audio_path: Path, whisper_model: str
) -> Optional[list[dict[str, Any]]]:
    try:
        import whisper
    except Exception:
        print(
            "Whisper fallback requested but the 'whisper' package is not installed. "
            "Install with: pip install -U openai-whisper"
        )
        return None

    try:
        device = _whisper_device()
        print(f"[whisper] Loading model '{whisper_model}' on device: {device}")
        model = whisper.load_model(whisper_model, device=device)
        result = model.transcribe(str(audio_path), verbose=False, fp16=(device != "cpu"))
        raw_segments = result.get("segments", [])
        segments = []
        for item in raw_segments:
            segments.append(
                {
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", 0.0)),
                    "text": item.get("text", ""),
                }
            )
        return normalize_segments(segments) or None
    except Exception as exc:
        print(f"Whisper failed: {exc}")
        return None


def get_video_duration_seconds(video_path: Path) -> Optional[float]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps > 0 and count > 0:
        return count / fps
    return None


def build_time_windows(
    segments: list[dict[str, Any]],
    window_seconds: int,
    duration_seconds: Optional[float],
) -> list[dict[str, Any]]:
    if not segments:
        return []

    max_segment_end = max(item["end"] for item in segments)
    total_duration = max(max_segment_end, duration_seconds or 0.0)
    if total_duration <= 0:
        total_duration = max_segment_end

    windows = []
    window_count = max(1, math.ceil(total_duration / window_seconds))
    for idx in range(window_count):
        start = idx * window_seconds
        end = min(total_duration, start + window_seconds)
        if end <= start:
            end = start + window_seconds

        overlap = [
            item["text"]
            for item in segments
            if item["start"] < end and item["end"] > start
        ]
        text = clean_transcript_text(" ".join(overlap))
        windows.append(
            {
                "index": idx,
                "start": start,
                "end": end,
                "transcript": text,
                "frame_file": None,
            }
        )
    return windows


def extract_midpoint_frames(
    video_path: Path,
    windows: list[dict[str, Any]],
    frames_dir: Path,
    frame_width: int,
) -> None:
    if cv2 is None:
        print("OpenCV is not installed. Skipping frame extraction.")
        return

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video for frame extraction.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None

    for window in windows:
        midpoint = (window["start"] + window["end"]) / 2.0
        if duration is not None:
            midpoint = min(max(0.0, midpoint), max(0.0, duration - 0.05))

        if fps > 0:
            target_frame = int(midpoint * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, midpoint * 1000.0)

        ok, frame = cap.read()
        if not ok:
            continue

        if frame_width > 0:
            original_h, original_w = frame.shape[:2]
            if original_w > frame_width:
                ratio = frame_width / float(original_w)
                resized_h = max(1, int(original_h * ratio))
                frame = cv2.resize(frame, (frame_width, resized_h))

        frame_file = frames_dir / f"window_{window['index']:04d}.jpg"
        cv2.imwrite(str(frame_file), frame)
        window["frame_file"] = frame_file.name

    cap.release()


def extract_keyframes_every_x_seconds(
    video_path: Path,
    output_dir: Path,
    every_seconds: int,
    frame_width: int,
) -> list[Path]:
    if cv2 is None:
        print("OpenCV is not installed. Skipping keyframe extraction.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video for keyframe extraction.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if duration <= 0:
        cap.release()
        return []

    saved = []
    ts = 0.0
    idx = 0
    while ts < duration:
        if fps > 0:
            target_frame = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        else:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if ok:
            if frame_width > 0:
                original_h, original_w = frame.shape[:2]
                if original_w > frame_width:
                    ratio = frame_width / float(original_w)
                    resized_h = max(1, int(original_h * ratio))
                    frame = cv2.resize(frame, (frame_width, resized_h))
            output_path = output_dir / f"keyframe_{idx:04d}_{int(ts):06d}s.jpg"
            cv2.imwrite(str(output_path), frame)
            saved.append(output_path)
            idx += 1
        ts += max(1, every_seconds)

    cap.release()
    return saved


def extract_keyframes_scene_change(
    video_path: Path,
    output_dir: Path,
    threshold: float,
    min_gap_sec: float,
    frame_width: int,
) -> list[Path]:
    """Save a frame only when mean pixel difference from the last saved frame exceeds threshold.

    Ported from video_extractor.py. Answers "what changed on screen?" rather than sampling
    blindly at fixed intervals — useful when you want exactly one frame per distinct UI state.
    """
    if cv2 is None:
        print("OpenCV is not installed. Skipping keyframe extraction.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video for scene-change keyframe extraction.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    min_gap_frames = int(fps * min_gap_sec) if fps > 0 else 1

    saved = []
    prev_gray = None
    last_saved_idx = -min_gap_frames
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gap_ok = (frame_idx - last_saved_idx) >= min_gap_frames

        if prev_gray is None or (gap_ok and cv2.absdiff(gray, prev_gray).mean() >= threshold):
            ts = frame_idx / fps if fps > 0 else frame_idx
            if frame_width > 0:
                original_h, original_w = frame.shape[:2]
                if original_w > frame_width:
                    ratio = frame_width / float(original_w)
                    frame = cv2.resize(frame, (frame_width, max(1, int(original_h * ratio))))
            output_path = output_dir / f"keyframe_{saved_count:04d}_{int(ts):06d}s.jpg"
            cv2.imwrite(str(output_path), frame)
            saved.append(output_path)
            prev_gray = gray
            last_saved_idx = frame_idx
            saved_count += 1

        frame_idx += 1

    cap.release()
    return saved


def extract_keyframes_ocr_change(
    video_path: Path,
    output_dir: Path,
    ocr_engine,
    similarity_threshold: float,
    min_gap_sec: float,
    frame_width: int,
    phash_threshold: int = 5,
) -> tuple[list[Path], dict[str, str]]:
    """Save a frame only when on-screen text changes significantly.

    Two-stage filter: a cheap dHash prefilter skips OCR on visually-similar frames
    (Hamming <= phash_threshold vs. last saved frame), then RapidOCR runs on
    survivors and a text-similarity check gates the final save.
    """
    if cv2 is None:
        print("OpenCV is not installed. Skipping OCR keyframe extraction.")
        return [], {}

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video for OCR keyframe extraction.")
        return [], {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # Sample at ~1 FPS to bound work regardless of source frame rate
    process_interval = int(round(fps)) if fps > 0 else 1

    saved: list[Path] = []
    keyframe_ocr: dict[str, str] = {}
    last_text = ""
    last_saved_hash: np.ndarray | None = None
    last_saved_ts = -min_gap_sec
    frame_idx = 0
    saved_count = 0
    ocr_calls = 0
    phash_skips = 0

    print(
        f"[ocr-filter] Starting OCR-driven extraction "
        f"(text_threshold={similarity_threshold}, phash_threshold={phash_threshold})..."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = frame_idx / fps if fps > 0 else frame_idx
        if frame_idx % process_interval == 0 and (ts - last_saved_ts) >= min_gap_sec:
            current_hash = compute_dhash(frame)

            if last_saved_hash is not None and hamming_distance(current_hash, last_saved_hash) <= phash_threshold:
                phash_skips += 1
                frame_idx += 1
                continue

            ocr_result, _elapsed = ocr_engine(frame)
            ocr_calls += 1
            current_text_lines = [item[1] for item in ocr_result] if ocr_result else []
            current_text = "\n".join(current_text_lines).strip()

            if not last_text:
                similarity = 0.0
            else:
                similarity = difflib.SequenceMatcher(None, last_text, current_text).ratio()

            if not last_text or similarity < similarity_threshold:
                if frame_width > 0:
                    original_h, original_w = frame.shape[:2]
                    if original_w > frame_width:
                        ratio = frame_width / float(original_w)
                        frame = cv2.resize(frame, (frame_width, max(1, int(original_h * ratio))))

                output_path = output_dir / f"keyframe_{saved_count:04d}_{int(ts):06d}s.jpg"
                cv2.imwrite(str(output_path), frame)
                saved.append(output_path)
                keyframe_ocr[output_path.name] = current_text

                last_text = current_text
                last_saved_hash = current_hash
                last_saved_ts = ts
                saved_count += 1
                print(f"[ocr-filter] Saved keyframe {saved_count} at {int(ts)}s (text_sim={similarity:.2f})")

        frame_idx += 1

    cap.release()
    print(
        f"[ocr-filter] Done: {saved_count} keyframes saved, "
        f"{ocr_calls} OCR calls, {phash_skips} phash skips."
    )
    return saved, keyframe_ocr


def extract_windowed_macro_chunk_frames(
    video_path: Path,
    windows: list[dict[str, Any]],
    output_dir: Path,
    capture_fps: int,
    max_frames_per_window: int,
    frame_width: int,
) -> None:
    if cv2 is None:
        print("OpenCV is not installed. Skipping macro-chunk frame extraction.")
        return
    if not windows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for window in windows:
        window["frame_files"] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video for macro-chunk frame extraction.")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if native_fps <= 0:
        native_fps = 30.0
    frame_interval = max(1, int(round(native_fps / max(1, capture_fps))))

    window_seconds = max(1.0, windows[0]["end"] - windows[0]["start"])
    frame_index = 0
    sampled_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_interval == 0:
            timestamp = frame_index / native_fps
            window_idx = int(timestamp // window_seconds)
            if 0 <= window_idx < len(windows):
                if frame_width > 0:
                    original_h, original_w = frame.shape[:2]
                    if original_w > frame_width:
                        ratio = frame_width / float(original_w)
                        resized_h = max(1, int(original_h * ratio))
                        frame = cv2.resize(frame, (frame_width, resized_h))

                file_name = f"window_{window_idx:04d}_sample_{sampled_index:06d}.jpg"
                output_path = output_dir / file_name
                cv2.imwrite(str(output_path), frame)
                windows[window_idx]["frame_files"].append(file_name)
                sampled_index += 1
        frame_index += 1

    cap.release()

    for window in windows:
        all_files = [output_dir / name for name in window["frame_files"]]
        selected = select_diverse_frames(all_files, max_frames_per_window)
        selected_names = {path.name for path in selected}
        for path in all_files:
            if path.name not in selected_names:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        window["frame_files"] = [path.name for path in selected]
        if window["frame_files"] and not window.get("frame_file"):
            window["frame_file"] = window["frame_files"][0]


def sample_keyframes(keyframe_paths: list[Path], max_frames: int) -> list[Path]:
    indices = sample_evenly_spaced_indices(len(keyframe_paths), max_frames)
    if len(indices) == len(keyframe_paths):
        return keyframe_paths
    selected = [keyframe_paths[idx] for idx in indices]
    deduped = []
    seen = set()
    for path in selected:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def select_diverse_frames(paths: list[Path], max_frames: int, min_diff: float = 4.0) -> list[Path]:
    """Return up to max_frames paths that are visually distinct from each other.

    Uses sequential scene-change comparison on small grayscale thumbnails so
    near-duplicate frames within the same window are skipped.
    """
    if not paths or max_frames <= 0:
        return []
    if cv2 is None or len(paths) <= max_frames:
        return paths

    thumb_size = (64, 36)

    def load_thumb(p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, thumb_size) if img is not None else None

    selected: list[Path] = []
    prev_thumb = None

    for p in paths:
        if len(selected) >= max_frames:
            break
        thumb = load_thumb(p)
        if thumb is None:
            continue
        if prev_thumb is None or cv2.absdiff(thumb, prev_thumb).mean() >= min_diff:
            selected.append(p)
            prev_thumb = thumb

    return selected


def sample_evenly_spaced_indices(total_items: int, max_items: int) -> list[int]:
    if total_items <= 0:
        return []
    if max_items <= 0 or total_items <= max_items:
        return list(range(total_items))
    if max_items == 1:
        return [total_items // 2]

    selected = []
    last_index = total_items - 1
    for i in range(max_items):
        selected.append(round(i * last_index / (max_items - 1)))

    deduped = []
    seen = set()
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    if len(deduped) >= max_items:
        return deduped[:max_items]

    for idx in range(total_items):
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
        if len(deduped) >= max_items:
            break
    return deduped


def analyze_transcript_only_vision_fallback(
    transcript: str,
    engine,
    video_url: str,
    failure_reason: str,
) -> str:
    trimmed = transcript[:20000]
    prompt = f"""
Vision analysis failed for this run. Build a transcript-only version of the report.

Video URL: {video_url}
Failure reason: {failure_reason}

Return a concise markdown report with exactly these headings:
1. # Full Demo Summary
2. ## Key Entities
3. ## End-to-End Workflow
4. ## PM View
5. ## UX View
6. ## Developer View
7. ## Unknowns / Assumptions

Use only transcript evidence and mark assumptions explicitly.

Transcript:
{trimmed}
"""
    return call_qwen(engine, prompt, [])


def analyze_macro_chunk_transcript_fallback(
    video_url: str,
    engine,
    window: dict[str, Any],
    failure_reason: str,
) -> str:
    transcript_text = window.get("transcript") or "[No transcript in this interval]"
    start_label = seconds_to_hhmmss(window["start"])
    end_label = seconds_to_hhmmss(window["end"])
    prompt = f"""
Vision analysis failed for this chunk. Build a transcript-only chunk summary.

Video URL: {video_url}
Window: {start_label} - {end_label}
Failure reason: {failure_reason}
Transcript snippet: {transcript_text}

Respond in markdown with:
- **Observed UI actions**
- **Feature or capability shown**
- **User goal in this step**
- **Entity/data objects involved**
- **UX notes**
"""
    return call_qwen(engine, prompt, [])


def analyze_with_qwen_vision(
    transcript: str,
    keyframe_paths: list[Path],
    engine,
    video_url: str,
    ocr_text: str = "",
) -> str:
    transcript_trimmed = transcript[:12000]

    ocr_section = (
        f"\nOn-screen text extracted via OCR from keyframes:\n{ocr_text[:8000]}\n"
        if ocr_text.strip() else ""
    )

    prompt = f"""
You are reviewing a construction software demo.

Inputs:
- Video URL: {video_url}
- Transcript text (from Whisper/captions)
- Chronological keyframes sampled from the video
{ocr_section}
Return a concise, practical markdown report with these headings:
1. # Full Demo Summary
2. ## Key Entities
3. ## End-to-End Workflow
4. ## PM View
5. ## UX View
6. ## Developer View
7. ## Unknowns / Assumptions

Use visuals, transcript, and OCR evidence together. If uncertain, mark as assumption.

Transcript:
{transcript_trimmed}
"""
    return call_qwen(engine, prompt, keyframe_paths)


def analyze_macro_chunk_window_with_vision(
    video_url: str,
    engine,
    frames_dir: Path,
    window: dict[str, Any],
) -> str:
    frame_files = window.get("frame_files", [])
    image_paths = []
    for name in frame_files:
        path = frames_dir / name
        if path.exists():
            image_paths.append(path)

    transcript_text = window.get("transcript") or "[No transcript in this interval]"
    ocr_text = window.get("ocr_text", "").strip()
    ocr_section = f"\nOn-screen text (OCR): {ocr_text}" if ocr_text else ""
    start_label = seconds_to_hhmmss(window["start"])
    end_label = seconds_to_hhmmss(window["end"])
    prompt = f"""
You are analyzing one workflow chunk from a construction software demo.

Video URL: {video_url}
Window: {start_label} - {end_label}
Transcript snippet: {transcript_text}{ocr_section}

Use the chronological images, transcript, and OCR text together.
Respond in markdown with:
- **Observed UI actions**
- **Feature or capability shown**
- **User goal in this step**
- **Entity/data objects involved**
- **UX notes**
"""
    return call_qwen(engine, prompt, image_paths)


def compile_macro_chunk_full_summary(
    video_url: str,
    engine,
    chunk_analyses: list[dict[str, Any]],
) -> str:
    timeline_blocks = []
    for chunk in chunk_analyses:
        timeline_blocks.append(
            f"[{seconds_to_hhmmss(chunk['start'])}-{seconds_to_hhmmss(chunk['end'])}]\n{chunk['analysis']}"
        )
    joined = "\n\n".join(timeline_blocks)[:50000]
    prompt = f"""
Based on the windowed workflow analyses below, produce a full construction software demo report.

Video URL: {video_url}

Format:
# Product Demo Executive Summary
## Key Features
## Key Entities
## End-to-End Workflow
## PM Insights
## UX Insights
## Developer / Architecture Insights
## Open Questions

Windowed analyses:
{joined}
"""
    return call_qwen(engine, prompt, [])


def write_macro_chunk_report(
    video_id: str,
    video_url: str,
    reports_dir: Path,
    relative_frame_dir: str,
    chunk_analyses: list[dict[str, Any]],
) -> Path:
    lines = []
    lines.append("# Windowed Macro-Chunk Analysis")
    lines.append("")
    lines.append(f"Source: {video_url}")
    lines.append("")
    for chunk in chunk_analyses:
        lines.append(
            f"## {seconds_to_hhmmss(chunk['start'])} - {seconds_to_hhmmss(chunk['end'])}"
        )
        lines.append("")
        for frame_name in chunk.get("frame_files", []):
            lines.append(f"![frame]({relative_frame_dir}/{frame_name})")
        if chunk.get("frame_files"):
            lines.append("")
        transcript_text = chunk.get("transcript") or "[No transcript text]"
        lines.append(f"**Transcript:** {transcript_text}")
        lines.append("")
        lines.append(chunk.get("analysis", "No analysis."))
        lines.append("")

    output = reports_dir / f"{video_id}_macro_chunk_analysis.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def build_timeline_markdown(
    video_url: str,
    windows: list[dict[str, Any]],
    relative_frame_dir: str,
) -> str:
    lines = []
    lines.append("# Transcript + Frame Timeline")
    lines.append("")
    lines.append(f"Source: {video_url}")
    lines.append("")
    for window in windows:
        start_label = seconds_to_hhmmss(window["start"])
        end_label = seconds_to_hhmmss(window["end"])
        lines.append(f"## {start_label} - {end_label}")
        lines.append("")
        if window.get("frame_file"):
            frame_rel = f"{relative_frame_dir}/{window['frame_file']}"
            lines.append(f"![{start_label}]({frame_rel})")
            lines.append("")
        transcript = window.get("transcript") or "[No transcript text in this interval]"
        lines.append(transcript)
        lines.append("")
        ocr = window.get("ocr_text", "").strip()
        if ocr:
            lines.append(f"**On-screen text:** {ocr}")
            lines.append("")
    return "\n".join(lines)


def build_timeline_html(
    video_url: str,
    windows: list[dict[str, Any]],
    relative_frame_dir: str,
) -> str:
    has_ocr = any(w.get("ocr_text", "").strip() for w in windows)
    rows = []
    for window in windows:
        start_label = seconds_to_hhmmss(window["start"])
        end_label = seconds_to_hhmmss(window["end"])
        transcript = html.escape(window.get("transcript") or "[No transcript text in this interval]")
        if window.get("frame_file"):
            frame_src = f"{relative_frame_dir}/{window['frame_file']}"
            frame_html = f'<img src="{frame_src}" alt="{start_label}" loading="lazy" />'
        else:
            frame_html = "<div class='missing'>No frame</div>"
        ocr_cell = (
            f"<td class='text'>{html.escape(window.get('ocr_text', '').strip())}</td>"
            if has_ocr else ""
        )
        rows.append(
            "<tr>"
            f"<td class='time'>{start_label} - {end_label}</td>"
            f"<td class='frame'>{frame_html}</td>"
            f"<td class='text'>{transcript}</td>"
            f"{ocr_cell}"
            "</tr>"
        )
    ocr_header = "<th>On-Screen Text (OCR)</th>" if has_ocr else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Transcript + Frame Timeline</title>
  <style>
    body {{
      margin: 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.4;
      color: #1f2937;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .source {{
      margin: 0 0 20px;
      color: #4b5563;
      word-break: break-all;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 10px;
      vertical-align: top;
    }}
    th {{
      text-align: left;
      background: #f3f4f6;
      font-weight: 600;
    }}
    .time {{
      width: 160px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
    }}
    .frame {{
      width: 380px;
    }}
    .frame img {{
      width: 100%;
      max-width: 360px;
      height: auto;
      border: 1px solid #d1d5db;
      display: block;
    }}
    .missing {{
      color: #6b7280;
      font-style: italic;
    }}
    .text {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    @media (max-width: 980px) {{
      .frame {{
        width: 230px;
      }}
    }}
  </style>
</head>
<body>
  <h1>Transcript + Frame Timeline</h1>
  <p class="source">Source: {html.escape(video_url)}</p>
  <table>
    <thead>
      <tr>
        <th>Time Window</th>
        <th>Frame</th>
        <th>Transcript</th>
        {ocr_header}
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def analyze_transcript_with_qwen(
    transcript: str,
    engine,
    video_url: str,
) -> str:
    trimmed = transcript[:45000]
    prompt = f"""You are analyzing a video transcript (software demo or industry discussion).

Video URL: {video_url}

Produce a practical analysis for a Product Manager, UX Designer, and Developer.
Use only transcript evidence. If uncertain, mark assumptions clearly.
Write in markdown using these exact section headings:

**Summary**
1-2 paragraph plain-language overview of what the product does and who it's for.

**Industry Context & Pain Points**
- What specific industry problems or "pain points" are discussed?
- Who is the primary target audience?
- What are the core market drivers mentioned?

**Key Entities**
For each important object in the product (data, UI, business concept), write:
- **Entity name**
  - Type: business / process / data / ui
  - Description: what it represents
  - Key attributes: list of fields or properties
  - Relationships: how it connects to other entities

**Workflow**
Number each step of any demo or process shown. For each:
- Stage name
  - Actor: who does this
  - Goal: why this step exists
  - Actions: what the user does
  - Inputs: what is needed
  - Outputs: what is produced
  - Risks or friction: observed issues or manual effort

**Product Manager View**
- Core Value Props
- Adoption Blockers
- Opportunity Areas

**UX Designer View**
- Strengths
- Usability Issues
- Recommended Improvements

**Developer View**
- Likely Modules
- Integration Points
- Data Model Hints
- Engineering Risks

**Open Questions**
List unanswered questions about the product, users, or implementation.

**Automation Opportunities**
For each manual step that could be automated:
- Manual step: what the user does by hand
- Current friction: why this is slow or error-prone
- Automation potential: what could replace the manual step
- Connected entities: what data objects this automation would link or update

Transcript:
{trimmed}
"""

    return call_qwen(engine, prompt, [])


def json_or_none(text: str) -> Optional[dict[str, Any]]:
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def analysis_to_markdown(video_url: str, analysis: dict[str, Any]) -> str:
    lines = []
    lines.append("# Construction Demo Analysis")
    lines.append("")
    lines.append(f"Source: {video_url}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(analysis.get("summary", "No summary available."))
    lines.append("")

    lines.append("## Key Entities")
    lines.append("")
    entities = analysis.get("key_entities", [])
    if not entities:
        lines.append("- No entities extracted.")
    else:
        for entity in entities:
            lines.append(f"- {entity.get('name', 'Unknown')}: {entity.get('description', '')}")

    lines.append("")
    lines.append("## Workflow")
    lines.append("")
    workflow = analysis.get("workflow", [])
    if not workflow:
        lines.append("- No workflow extracted.")
    else:
        for idx, stage in enumerate(workflow, start=1):
            lines.append(
                f"{idx}. {stage.get('stage', 'Stage')} | Actor: {stage.get('actor', '')} | Goal: {stage.get('goal', '')}"
            )

    lines.append("")
    lines.append("## PM View")
    lines.append("")
    pm_view = analysis.get("pm_view", {})
    lines.append(f"- Core Value Props: {', '.join(pm_view.get('core_value_props', []))}")
    lines.append(f"- Adoption Blockers: {', '.join(pm_view.get('adoption_blockers', []))}")
    lines.append(f"- Opportunity Areas: {', '.join(pm_view.get('opportunity_areas', []))}")

    lines.append("")
    lines.append("## UX View")
    lines.append("")
    ux_view = analysis.get("ux_view", {})
    lines.append(f"- Strengths: {', '.join(ux_view.get('strengths', []))}")
    lines.append(f"- Usability Issues: {', '.join(ux_view.get('usability_issues', []))}")
    lines.append(f"- Recommended Improvements: {', '.join(ux_view.get('recommended_improvements', []))}")

    lines.append("")
    lines.append("## Developer View")
    lines.append("")
    dev_view = analysis.get("developer_view", {})
    lines.append(f"- Likely Modules: {', '.join(dev_view.get('likely_modules', []))}")
    lines.append(f"- Integration Points: {', '.join(dev_view.get('integration_points', []))}")
    lines.append(f"- Data Model Hints: {', '.join(dev_view.get('data_model_hints', []))}")
    lines.append(f"- Engineering Risks: {', '.join(dev_view.get('engineering_risks', []))}")

    lines.append("")
    lines.append("## Open Questions")
    lines.append("")
    for item in analysis.get("open_questions", []):
        lines.append(f"- {item}")

    automation = analysis.get("automation_opportunities", [])
    if automation:
        lines.append("")
        lines.append("## Automation Opportunities")
        lines.append("")
        for idx, item in enumerate(automation, start=1):
            lines.append(f"### {idx}. {item.get('manual_step', 'Manual step')}")
            lines.append(f"- **Current friction:** {item.get('current_friction', '')}")
            lines.append(f"- **Automation potential:** {item.get('automation_potential', '')}")
            connected = item.get("connected_entities", [])
            if connected:
                lines.append(f"- **Connected entities:** {', '.join(connected)}")
            lines.append("")

    return "\n".join(lines)


def assign_keyframes_to_windows(
    windows: list[dict[str, Any]],
    keyframes: list[Path],
    keyframe_ocr: Optional[dict[str, str]] = None,
) -> None:
    """Set window['frame_file'] (and optionally 'ocr_text') to the keyframe closest to each window's midpoint."""
    for window in windows:
        midpoint = (window["start"] + window["end"]) / 2.0
        best: Optional[Path] = None
        best_dist = float("inf")
        for kf in keyframes:
            try:
                ts = float(kf.stem.split("_")[-1].rstrip("s"))
            except (ValueError, IndexError):
                continue
            dist = abs(ts - midpoint)
            if dist < best_dist:
                best_dist = dist
                best = kf
        if best is not None:
            window["frame_file"] = best.name
            if keyframe_ocr is not None:
                window["ocr_text"] = keyframe_ocr.get(best.name, "")


def write_timeline_outputs(
    video_id: str,
    video_url: str,
    windows: list[dict[str, Any]],
    reports_dir: Path,
    frames_dir: Path,
) -> None:
    import os as _os
    relative_frame_dir = _os.path.relpath(str(frames_dir), str(reports_dir))
    timeline_md = build_timeline_markdown(video_url, windows, relative_frame_dir)
    timeline_html = build_timeline_html(video_url, windows, relative_frame_dir)

    (reports_dir / f"{video_id}_timeline.md").write_text(timeline_md, encoding="utf-8")
    (reports_dir / f"{video_id}_timeline.html").write_text(timeline_html, encoding="utf-8")
    (reports_dir / f"{video_id}_timeline.json").write_text(
        json.dumps(windows, indent=2), encoding="utf-8"
    )
    if not frames_dir.exists():
        print(f"[{video_id}] Timeline created without frame images.")


def run(
    video_urls: list[str],
    model: str,
    vision_model: str,
    work_dir: Path,
    download: bool,
    run_analysis: bool,
    run_vision_summary: bool,
    generate_timeline: bool,
    timeline_window_seconds: int,
    keyframe_seconds: int,
    frame_width: int,
    max_vision_frames: int,
    capture_fps: int,
    macro_window_seconds: int,
    macro_frames_per_window: int,
    run_macro_chunking: bool,
    use_whisper_fallback: bool,
    whisper_model: str,
    keyframe_mode: str = "interval",
    scene_threshold: float = 25.0,
    min_scene_gap: float = 1.0,
    video_files: list[str] | None = None,
    ocr_similarity_threshold: float = 0.85,
    phash_threshold: int = 5,
    quantization: str | None = None,
) -> None:
    reports_dir = work_dir / "reports"
    frames_root = work_dir / "frames"
    reports_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    qwen_engine = None
    if run_analysis or run_vision_summary or run_macro_chunking:
        qwen_engine = build_qwen_engine(vision_model, quantization=quantization)

    ocr_engine = None
    if keyframe_mode == "ocr":
        ocr_engine = build_ocr_engine()

    # Build unified input list: each entry is (url_or_none, local_file_or_none).
    # --url alone:        (url, None)   — download + full transcript chain
    # --file alone:       (None, file)  — local video, Whisper only
    # --url + --file 1:1: (url, file)   — local video, URL used for transcript lookup
    inputs: list[tuple[str | None, Path | None]] = []
    files = [Path(f) for f in (video_files or [])]
    if video_urls and files:
        if len(video_urls) != len(files):
            print(
                f"[error] --url and --file counts must match when both are provided "
                f"({len(video_urls)} URLs, {len(files)} files). Exiting."
            )
            return
        inputs = list(zip(video_urls, files))
    elif video_urls:
        inputs = [(u, None) for u in video_urls]
    elif files:
        inputs = [(None, f) for f in files]
    else:
        print("[error] No --url or --file inputs provided.")
        return

    for url, local_file in inputs:
        # Determine video_id
        if url:
            try:
                video_id = extract_video_id(url)
            except Exception:
                if local_file:
                    video_id = local_file.stem
                else:
                    print(f"[skip] Could not parse video ID from URL: {url}")
                    continue
        else:
            video_id = local_file.stem  # type: ignore[union-attr]

        # Non-None label for report headers and LLM prompts (file-only runs have url=None).
        # Distinct from `url`, which stays None when there's no remote source so download /
        # transcript-API code paths correctly skip themselves.
        source_label = url or (str(local_file) if local_file else video_id)

        # Determine video_path — use local file if provided, else download
        video_path: Path | None = None
        if local_file:
            if local_file.exists():
                video_path = local_file
                print(f"[{video_id}] Using local file: {local_file}")
            else:
                print(f"[{video_id}] Local file not found: {local_file}. Skipping.")
                continue
        else:
            require_video = (
                download
                or generate_timeline
                or use_whisper_fallback
                or run_vision_summary
                or run_macro_chunking
            )
            if require_video:
                print(f"[{video_id}] Downloading video...")
                video_path = download_video(url, video_id, work_dir)
                if video_path:
                    print(f"[{video_id}] Video saved: {video_path}")
                else:
                    print(f"[{video_id}] Video download failed.")

        # Transcript acquisition
        # Local-file-only: skip YouTube API and yt-dlp (no URL source), go straight to Whisper.
        # URL provided: full fallback chain — API → yt-dlp subtitles → Whisper.
        segments = None
        transcript_source = ""
        if url:
            print(f"[{video_id}] Extracting transcript from YouTube transcript API...")
            segments = get_transcript_segments_from_api(video_id)
            transcript_source = "youtube_transcript_api"

            if not segments:
                print(f"[{video_id}] API transcript unavailable. Trying yt-dlp subtitles...")
                segments = get_transcript_segments_from_ytdlp(url, work_dir)
                transcript_source = "yt_dlp_subtitles"
        else:
            print(f"[{video_id}] No URL provided — skipping YouTube transcript sources.")

        if not segments and use_whisper_fallback:
            if video_path is None and url:
                print(f"[{video_id}] Downloading video for Whisper fallback...")
                video_path = download_video(url, video_id, work_dir)
            if video_path is not None:
                audio_path = extract_audio_for_whisper(
                    video_path, work_dir / "audio", video_id
                )
                print(f"[{video_id}] Running Whisper fallback ({whisper_model})...")
                if audio_path is not None:
                    segments = transcribe_with_whisper_audio(audio_path, whisper_model)
                else:
                    segments = transcribe_with_whisper(video_path, whisper_model)
                transcript_source = f"whisper_{whisper_model}"
        elif not segments and not use_whisper_fallback and not url:
            print(
                f"[{video_id}] No URL and Whisper fallback is disabled — cannot get transcript."
            )

        if not segments:
            print(f"[{video_id}] Could not get transcript from any source.")
            continue

        print(f"[{video_id}] Transcript source: {transcript_source} ({len(segments)} segments)")
        full_transcript = join_segments_text(segments)

        # Extract keyframes once — shared by timeline and vision summary
        keyframes: list[Path] = []
        keyframe_ocr: dict[str, str] = {}
        keyframes_dir = frames_root / video_id / "keyframes"
        if (generate_timeline or run_vision_summary) and video_path is not None:
            if keyframe_mode == "ocr":
                keyframes, keyframe_ocr = extract_keyframes_ocr_change(
                    video_path=video_path,
                    output_dir=keyframes_dir,
                    ocr_engine=ocr_engine,
                    similarity_threshold=ocr_similarity_threshold,
                    min_gap_sec=min_scene_gap,
                    frame_width=frame_width,
                    phash_threshold=phash_threshold,
                )
            elif keyframe_mode == "scene":
                print(
                    f"[{video_id}] Extracting keyframes on scene change "
                    f"(threshold={scene_threshold}, min_gap={min_scene_gap}s)..."
                )
                keyframes = extract_keyframes_scene_change(
                    video_path=video_path,
                    output_dir=keyframes_dir,
                    threshold=scene_threshold,
                    min_gap_sec=min_scene_gap,
                    frame_width=frame_width,
                )
            else:
                print(f"[{video_id}] Extracting keyframes every {keyframe_seconds}s...")
                keyframes = extract_keyframes_every_x_seconds(
                    video_path=video_path,
                    output_dir=keyframes_dir,
                    every_seconds=keyframe_seconds,
                    frame_width=frame_width,
                )
            print(f"[{video_id}] {len(keyframes)} keyframes saved to {keyframes_dir}")

        if generate_timeline:
            duration = get_video_duration_seconds(video_path) if video_path else None
            windows = build_time_windows(segments, timeline_window_seconds, duration)
            if keyframes:
                assign_keyframes_to_windows(windows, keyframes, keyframe_ocr if keyframe_ocr else None)
            elif video_path is None:
                print(f"[{video_id}] No video available. Timeline will be text-only.")
            write_timeline_outputs(video_id, source_label, windows, reports_dir, keyframes_dir)
            print(
                f"[{video_id}] Timeline files: "
                f"{reports_dir / f'{video_id}_timeline.html'} and {reports_dir / f'{video_id}_timeline.md'}"
            )

        if run_macro_chunking:
            if video_path is None and url:
                print(f"[{video_id}] Downloading video for macro-chunking...")
                video_path = download_video(url, video_id, work_dir)

            if video_path is None:
                print(f"[{video_id}] Skipping macro-chunking: no video available.")
            else:
                duration = get_video_duration_seconds(video_path)
                macro_windows = build_time_windows(segments, macro_window_seconds, duration)
                macro_frames_dir = frames_root / video_id / "macro_chunks"
                print(
                    f"[{video_id}] Extracting {capture_fps} fps samples in "
                    f"{macro_window_seconds}s windows..."
                )
                extract_windowed_macro_chunk_frames(
                    video_path=video_path,
                    windows=macro_windows,
                    output_dir=macro_frames_dir,
                    capture_fps=capture_fps,
                    max_frames_per_window=macro_frames_per_window,
                    frame_width=frame_width,
                )
                chunk_analyses: list[dict[str, Any]] = []
                for window in macro_windows:
                    if not window.get("frame_files"):
                        continue
                    try:
                        analysis = analyze_macro_chunk_window_with_vision(
                            video_url=source_label,
                            engine=qwen_engine,
                            frames_dir=macro_frames_dir,
                            window=window,
                        )
                    except Exception as exc:
                        print(
                            f"[{video_id}] Vision chunk failed "
                            f"({seconds_to_hhmmss(window['start'])}-{seconds_to_hhmmss(window['end'])}): {exc}"
                        )
                        print(
                            f"[{video_id}] Falling back to transcript-only chunk analysis "
                            f"for this window."
                        )
                        try:
                            analysis = analyze_macro_chunk_transcript_fallback(
                                video_url=source_label,
                                engine=qwen_engine,
                                window=window,
                                failure_reason=str(exc),
                            )
                        except Exception:
                            transcript_text = window.get("transcript") or "[No transcript for this window]"
                            analysis = f"**Transcript only** (Qwen unavailable):\n\n{transcript_text}"
                    chunk_analyses.append(
                        {
                            "start": window["start"],
                            "end": window["end"],
                            "transcript": window.get("transcript", ""),
                            "frame_files": window.get("frame_files", []),
                            "analysis": analysis,
                        }
                    )

                if chunk_analyses:
                    relative_frame_dir = str(Path("..") / "frames" / video_id / "macro_chunks")
                    macro_report = write_macro_chunk_report(
                        video_id=video_id,
                        video_url=source_label,
                        reports_dir=reports_dir,
                        relative_frame_dir=relative_frame_dir,
                        chunk_analyses=chunk_analyses,
                    )
                    print(f"[{video_id}] Macro-chunk analysis: {macro_report}")
                    try:
                        full_summary = compile_macro_chunk_full_summary(
                            video_url=source_label,
                            engine=qwen_engine,
                            chunk_analyses=chunk_analyses,
                        )
                    except Exception as exc:
                        print(f"[{video_id}] Macro-chunk summary failed: {exc}")
                    else:
                        macro_summary_output = reports_dir / f"{video_id}_macro_chunk_summary.md"
                        macro_summary_output.write_text(full_summary, encoding="utf-8")
                        print(f"[{video_id}] Macro-chunk summary: {macro_summary_output}")
                else:
                    print(f"[{video_id}] Macro-chunking extracted no usable frame windows.")

        if run_vision_summary:
            if not keyframes:
                print(f"[{video_id}] Skipping vision summary: no keyframes extracted.")
            else:
                selected_keyframes = sample_keyframes(keyframes, max_vision_frames)
                print(
                    f"[{video_id}] Running Native Qwen vision summary on {len(selected_keyframes)} keyframes..."
                )
                full_ocr_text = "\n\n".join(
                    t for t in keyframe_ocr.values() if t
                ) if keyframe_ocr else ""
                try:
                    summary = analyze_with_qwen_vision(
                        transcript=full_transcript,
                        keyframe_paths=selected_keyframes,
                        engine=qwen_engine,
                        video_url=source_label,
                        ocr_text=full_ocr_text,
                    )
                except Exception as exc:
                    print(f"[{video_id}] Vision summary failed: {exc}")
                    print(
                        f"[{video_id}] Falling back to transcript-only summary."
                    )
                    try:
                        summary = analyze_transcript_only_vision_fallback(
                            transcript=full_transcript,
                            engine=qwen_engine,
                            video_url=source_label,
                            failure_reason=str(exc),
                        )
                    except Exception as fallback_exc:
                        summary = (
                            "# Full Demo Summary\n\n"
                            "Vision summary failed and transcript-only fallback also failed.\n\n"
                            f"- Vision error: {exc}\n"
                            f"- Transcript fallback error: {fallback_exc}\n"
                        )
                vision_output = reports_dir / f"{video_id}_vision_summary.md"
                vision_output.write_text(summary, encoding="utf-8")
                print(f"[{video_id}] Vision summary report: {vision_output}")

        if run_analysis:
            print(f"[{video_id}] Running Native Qwen analysis...")
            try:
                raw = analyze_transcript_with_qwen(full_transcript, qwen_engine, source_label)
            except Exception as exc:
                print(f"[{video_id}] Analysis failed: {exc}")
                continue
            output = f"# Demo Analysis\n\nSource: {source_label}\n\n{raw}"
            output_file = reports_dir / f"{video_id}_analysis.md"
            output_file.write_text(output, encoding="utf-8")
            print(f"[{video_id}] Analysis report: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract YouTube frames + transcript timeline and generate PM/UX/dev analysis via Native Qwen2.5-VL."
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help=(
            "Video URL (YouTube or Wistia). Repeat for multiple videos. "
            "Use alone to download, or pair 1:1 with --file to use a local copy."
        ),
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Local video file path. Use alone (Whisper transcript only) or "
            "pair 1:1 with --url (URL used for transcript lookup, download skipped)."
        ),
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        help="Qwen model ID served by vLLM (default: Qwen/Qwen2.5-VL-32B-Instruct-AWQ).",
    )
    parser.add_argument(
        "--vision-model",
        default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        help="Qwen model ID for vision summary served by vLLM (default: Qwen/Qwen2.5-VL-32B-Instruct-AWQ).",
    )
    parser.add_argument(
        "--quantization",
        choices=["awq_marlin", "nvfp4", "fp8", "gptq_marlin", "none"],
        default=None,
        help=(
            "vLLM quantization override. Default (unset) lets vLLM auto-detect from the "
            "checkpoint config — AWQ checkpoints auto-select awq_marlin on Blackwell. "
            "Set to 'nvfp4' when using a pre-quantized NVFP4 checkpoint."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default="artifacts",
        help="Output directory for reports and assets (default: artifacts).",
    )
    parser.add_argument(
        "--download-video",
        action="store_true",
        help="Keep downloaded video file even if only timeline/report outputs are needed.",
    )
    parser.add_argument(
        "--analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable PM/UX/dev analysis output (default: enabled).",
    )
    parser.add_argument(
        "--vision-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable keyframe+transcript vision summary (default: enabled).",
    )
    parser.add_argument(
        "--macro-chunking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Windowed macro-chunking: 6 fps capture grouped into 15s windows, one LLM "
            "vision call per window. Off by default — the OCR-gated keyframe pipeline "
            "usually covers the same ground with one call. Enable for long-form videos "
            "(45+ min) or purely-visual content where per-window granularity matters."
        ),
    )
    parser.add_argument(
        "--timeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable transcript+frame timeline outputs (default: enabled).",
    )
    parser.add_argument(
        "--timeline-window-seconds",
        type=int,
        default=20,
        help="Seconds per timeline window (default: 20).",
    )
    parser.add_argument(
        "--keyframe-seconds",
        type=int,
        default=20,
        help="Extract one keyframe every X seconds for vision summary (default: 20).",
    )
    parser.add_argument(
        "--macro-window-seconds",
        type=int,
        default=15,
        help="Window size for macro-chunking in seconds (default: 15).",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=6,
        help="Frame sampling rate (fps) for macro-chunking (default: 6).",
    )
    parser.add_argument(
        "--macro-frames-per-window",
        type=int,
        default=8,
        help="Max selected frames per macro window sent to vision model (default: 8).",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=960,
        help="Resize timeline frame width in pixels (default: 960).",
    )
    parser.add_argument(
        "--max-vision-frames",
        type=int,
        default=24,
        help="Max keyframes to include in the vision summary (default: 24).",
    )
    parser.add_argument(
        "--whisper-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Whisper if transcript API and subtitle extraction fail (default: enabled).",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model for fallback transcription (default: base).",
    )
    parser.add_argument(
        "--keyframe-mode",
        choices=["interval", "scene", "ocr"],
        default="ocr",
        help=(
            "Keyframe extraction mode. "
            "'interval' saves one frame every --keyframe-seconds. "
            "'scene' saves a frame on visual pixel change. "
            "'ocr' uses a dHash prefilter + RapidOCR text-change detection (default)."
        ),
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=25.0,
        help="Mean pixel difference threshold for scene-change detection (default: 25).",
    )
    parser.add_argument(
        "--ocr-similarity-threshold",
        type=float,
        default=0.85,
        help="Text similarity threshold for OCR-based keyframing (default: 0.85).",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=5,
        help=(
            "Hamming-distance threshold (out of 64) for the dHash prefilter in ocr mode. "
            "Frames within this distance of the last saved frame skip OCR entirely (default: 5)."
        ),
    )
    parser.add_argument(
        "--min-keyframe-gap",
        type=float,
        default=1.0,
        help="Minimum seconds between saved frames (default: 1.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.url and not args.file:
        import sys
        print("error: at least one --url or --file argument is required.")
        sys.exit(1)
    run(
        video_urls=args.url,
        video_files=args.file or None,
        model=args.model,
        vision_model=args.vision_model,
        work_dir=Path(args.work_dir),
        download=args.download_video,
        run_analysis=args.analysis,
        run_vision_summary=args.vision_summary,
        generate_timeline=args.timeline,
        timeline_window_seconds=max(5, args.timeline_window_seconds),
        keyframe_seconds=max(5, args.keyframe_seconds),
        frame_width=max(240, args.frame_width),
        max_vision_frames=max(4, args.max_vision_frames),
        capture_fps=max(1, args.capture_fps),
        macro_window_seconds=max(5, args.macro_window_seconds),
        macro_frames_per_window=max(1, args.macro_frames_per_window),
        run_macro_chunking=args.macro_chunking,
        use_whisper_fallback=args.whisper_fallback,
        whisper_model=args.whisper_model,
        keyframe_mode=args.keyframe_mode,
        scene_threshold=args.scene_threshold,
        min_scene_gap=args.min_keyframe_gap,
        ocr_similarity_threshold=args.ocr_similarity_threshold,
        phash_threshold=args.phash_threshold,
        quantization=(None if args.quantization in (None, "none") else args.quantization),
    )
