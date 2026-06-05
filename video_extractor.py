import argparse
import html
import os
import re
import sys
import base64
import subprocess
import tempfile
import cv2
import whisper
import yt_dlp
from openai import OpenAI
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

FRAME_INTERVAL_SEC = 5   # extract one frame every N seconds


def extract_video_id(video_url: str) -> str:
    import hashlib
    parsed = urlparse(video_url)
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.lstrip("/").split("/")[0]
    if "youtube.com" in parsed.netloc:
        query_id = parse_qs(parsed.query).get("v")
        if query_id and query_id[0]:
            return query_id[0]
    if "wistia.com" in parsed.netloc:
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            return parts[-1]
    wvideo = parse_qs(parsed.query).get("wvideo")
    if wvideo and wvideo[0]:
        return wvideo[0]
    path_parts = [p for p in parsed.path.split("/") if p]
    if path_parts:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", path_parts[-1])[:40]
        if slug:
            return slug
    return hashlib.md5(video_url.encode()).hexdigest()[:12]

BACKENDS = {
    "openai": {
        "base_url": None,
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2-vision",
        "api_key_env": None,
    },
}


def build_client(backend: str) -> tuple[OpenAI, str]:
    cfg = BACKENDS[backend]
    if cfg["api_key_env"]:
        api_key = os.environ.get(cfg["api_key_env"])
        if not api_key:
            print(f"Error: {cfg['api_key_env']} environment variable not set.")
            sys.exit(1)
    else:
        api_key = "ollama"

    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]

    return OpenAI(**kwargs), cfg["model"]


def download_video(url: str, output_path: str) -> str:
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def extract_frames_interval(video_path: str, output_dir: Path, interval_sec: float) -> list[Path]:
    output_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, int(fps * interval_sec))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"Video: {total_frames} frames @ {fps:.1f} fps ({duration_sec:.1f}s)")
    print(f"Interval mode: one frame every {interval_sec}s")

    saved = []
    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps if fps > 0 else frame_idx
        out_path = output_dir / f"frame_{frame_idx:06d}_{ts:.1f}s.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(out_path)
        frame_idx += frame_step

    cap.release()
    print(f"Saved {len(saved)} frames to {output_dir}/")
    return saved


def extract_frames_scene(video_path: str, output_dir: Path, threshold: float, min_gap_sec: float) -> list[Path]:
    """Save a frame only when mean pixel difference from the last saved frame exceeds threshold."""
    output_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    min_gap_frames = int(fps * min_gap_sec)

    print(f"Video: {total_frames} frames @ {fps:.1f} fps ({duration_sec:.1f}s)")
    print(f"Scene mode: threshold={threshold}, min gap={min_gap_sec}s between saves")

    saved = []
    prev_gray = None
    last_saved_idx = -min_gap_frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gap_ok = (frame_idx - last_saved_idx) >= min_gap_frames

        if prev_gray is None or (gap_ok and cv2.absdiff(gray, prev_gray).mean() >= threshold):
            ts = frame_idx / fps if fps > 0 else frame_idx
            out_path = output_dir / f"frame_{frame_idx:06d}_{ts:.1f}s.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(out_path)
            prev_gray = gray
            last_saved_idx = frame_idx

        frame_idx += 1

    cap.release()
    print(f"Saved {len(saved)} frames to {output_dir}/")
    return saved


def extract_frames(video_path: str, output_dir: Path, interval_sec: float,
                   scene_threshold: float | None, min_gap_sec: float) -> list[Path]:
    if scene_threshold is not None:
        return extract_frames_scene(video_path, output_dir, scene_threshold, min_gap_sec)
    return extract_frames_interval(video_path, output_dir, interval_sec)


def extract_audio(video_path: str, audio_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path],
        check=True,
        capture_output=True,
    )


def transcribe_audio(audio_path: str, model_name: str) -> list[dict]:
    print(f"Loading Whisper model '{model_name}' ...")
    model = whisper.load_model(model_name)
    print("Transcribing audio ...")
    result = model.transcribe(audio_path, fp16=False)
    return result["segments"]


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_text_from_frame(client: OpenAI, model: str, frame_path: Path) -> str:
    b64 = encode_image(frame_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL visible text from this video frame exactly as it appears. "
                            "Include UI labels, headers, menu items, tooltips, data values, and any "
                            "on-screen text. Output only the extracted text, one item per line."
                        ),
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def timestamp_str(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def get_audio_snippet(segments: list[dict], frame_ts: float, window_sec: float = 5.0) -> str:
    """Return Whisper transcript text that overlaps the window around frame_ts."""
    lo, hi = frame_ts - window_sec / 2, frame_ts + window_sec / 2
    lines = [
        seg["text"].strip()
        for seg in segments
        if seg["end"] >= lo and seg["start"] <= hi and seg["text"].strip()
    ]
    return " ".join(lines)


def build_timeline_html(source_url: str, rows: list[dict]) -> str:
    """
    rows: list of {ts: float, frame_path: Path, ocr_text: str, audio_text: str}
    frame_path is relative to the HTML file location (project root).
    """
    has_audio = any(r["audio_text"] for r in rows)
    audio_col = "<th>Audio (Whisper)</th>" if has_audio else ""

    row_html_parts = []
    for r in rows:
        ts_label = html.escape(timestamp_str(r["ts"]))
        img_src = html.escape(str(r["frame_path"]))
        img_tag = f'<img src="{img_src}" alt="{ts_label}" loading="lazy" />'
        ocr = html.escape(r["ocr_text"] or "")
        audio_cell = f"<td class='text'>{html.escape(r['audio_text'])}</td>" if has_audio else ""
        row_html_parts.append(
            f"<tr>"
            f"<td class='time'>{ts_label}</td>"
            f"<td class='frame'>{img_tag}</td>"
            f"<td class='text'>{ocr}</td>"
            f"{audio_cell}"
            f"</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Video Timeline</title>
  <style>
    body {{
      margin: 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.4;
      color: #1f2937;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .source {{ margin: 0 0 20px; color: #4b5563; word-break: break-all; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ border: 1px solid #d1d5db; padding: 10px; vertical-align: top; }}
    th {{ text-align: left; background: #f3f4f6; font-weight: 600; }}
    .time {{ width: 90px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: nowrap; }}
    .frame {{ width: 360px; }}
    .frame img {{ width: 100%; max-width: 340px; height: auto; border: 1px solid #d1d5db; display: block; }}
    .text {{ white-space: pre-wrap; word-break: break-word; font-size: 13px; }}
    @media (max-width: 980px) {{ .frame {{ width: 200px; }} }}
  </style>
</head>
<body>
  <h1>Video Timeline</h1>
  <p class="source">Source: {html.escape(source_url)}</p>
  <table>
    <thead>
      <tr>
        <th>Time</th>
        <th>Frame</th>
        <th>OCR Text</th>
        {audio_col}
      </tr>
    </thead>
    <tbody>
      {"".join(row_html_parts)}
    </tbody>
  </table>
</body>
</html>
"""


def parse_frame_timestamp(frame_path: Path) -> float:
    """Extract the timestamp in seconds from a frame filename like frame_000150_5.0s.jpg"""
    try:
        return float(frame_path.stem.split("_")[-1].rstrip("s"))
    except (ValueError, IndexError):
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Extract frames and text from a video.")
    parser.add_argument(
        "--url",
        required=True,
        metavar="URL",
        help="Video URL to download and analyze (YouTube, Wistia, or other yt-dlp-supported URL)",
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="ollama",
        help="Vision backend for frame text extraction (default: ollama)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model for audio transcription (default: base)",
    )
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help="Skip audio transcription",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Skip frame extraction and vision text extraction",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=FRAME_INTERVAL_SEC,
        metavar="SEC",
        help=f"Seconds between frames in interval mode (default: {FRAME_INTERVAL_SEC})",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=None,
        metavar="0-255",
        help="Use scene-change detection instead of interval sampling. "
             "Saves a frame when mean pixel difference exceeds this value (e.g. 25). "
             "Lower = more sensitive.",
    )
    parser.add_argument(
        "--min-scene-gap",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Minimum seconds between scene-change saves (default: 1.0)",
    )
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    output_dir = Path(f"artifacts/frames/{video_id}/frames")
    text_output = Path(f"artifacts/transcripts/{video_id}.txt")
    html_output = Path(f"artifacts/reports/{video_id}_timeline.html")

    text_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.parent.mkdir(parents=True, exist_ok=True)

    client, model = (None, None)
    if not args.no_frames:
        client, model = build_client(args.backend)
        print(f"Using vision backend: {args.backend} (model: {model})")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path_template = os.path.join(tmpdir, "video.%(ext)s")
        print(f"Downloading video from {args.url} ...")
        actual_path = download_video(args.url, video_path_template)
        if not os.path.exists(actual_path):
            candidates = list(Path(tmpdir).glob("video.*"))
            if not candidates:
                print("Error: video download failed — no file found.")
                sys.exit(1)
            actual_path = str(candidates[0])
        print(f"Downloaded: {actual_path}")

        frames = []
        if not args.no_frames:
            frames = extract_frames(
                actual_path, output_dir,
                interval_sec=args.frame_interval,
                scene_threshold=args.scene_threshold,
                min_gap_sec=args.min_scene_gap,
            )

        segments = []
        if not args.no_whisper:
            audio_path = os.path.join(tmpdir, "audio.wav")
            print("Extracting audio ...")
            extract_audio(actual_path, audio_path)
            segments = transcribe_audio(audio_path, args.whisper_model)
            print(f"Transcribed {len(segments)} audio segments")

    # Collect per-frame OCR results
    timeline_rows = []
    with open(text_output, "w") as out:
        if segments:
            out.write("=== AUDIO TRANSCRIPT (Whisper) ===\n\n")
            for seg in segments:
                out.write(f"[{seg['start']:.1f}s – {seg['end']:.1f}s]  {seg['text'].strip()}\n")
            out.write("\n\n")

        if frames:
            out.write("=== FRAME TEXT (Vision OCR) ===\n\n")
            for i, frame_path in enumerate(frames, 1):
                print(f"[{i}/{len(frames)}] Extracting text from {frame_path.name} ...")
                try:
                    ocr_text = extract_text_from_frame(client, model, frame_path)
                except Exception as e:
                    ocr_text = f"[Error: {e}]"
                out.write(f"--- {frame_path.name} ---\n{ocr_text}\n\n")
                print(f"  -> {len(ocr_text)} chars extracted")

                ts = parse_frame_timestamp(frame_path)
                audio_snippet = get_audio_snippet(segments, ts) if segments else ""
                timeline_rows.append({
                    "ts": ts,
                    "frame_path": frame_path,
                    "ocr_text": ocr_text,
                    "audio_text": audio_snippet,
                })

    if timeline_rows:
        html_content = build_timeline_html(args.url, timeline_rows)
        html_output.write_text(html_content)
        print(f"Timeline saved to {html_output}")

    print(f"Text saved to {text_output}")
    if frames:
        print(f"Frames saved in {output_dir}/")


if __name__ == "__main__":
    main()
