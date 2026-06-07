"""Shared video-analysis pipeline: acquisition, transcripts, frames, reporting.

Backend-agnostic — contains no LLM calls. Both product_demo_video_analyzer.py
(Ollama) and product_demo_video_analyzer_dgx.py (Qwen/vLLM) import from here.
"""

import html
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from youtube_transcript_helpers import fetch_transcript_entries

try:
    import cv2
except Exception:
    cv2 = None


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
