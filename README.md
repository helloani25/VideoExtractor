# VideoExtractor: YouTube Demo Analysis Pipeline

This project contains two tools that answer different questions about a product demo video:

| Script | Question answered | Best for |
|---|---|---|
| `video_extractor.py` | **"What text is visible on screen?"** | Reading UI elements, labels, menus, and data values shown in the demo |
| `product_demo_video_analyzer.py` | **"What is the presenter doing and why?"** | Understanding workflow, features, and intent from the audio narrative |

Both can be combined: `product_demo_video_analyzer.py` supports `--ocr-backend` to answer both questions in a single run.

---

## `video_extractor.py` — Frame OCR Extractor

**Answers: "What text is visible on screen?"**

Useful when a demo shows UI elements, labels, menus, data values, or any on-screen content you want to read. Pass any video URL (YouTube, Wistia, or other yt-dlp-supported source). It extracts frames at a fixed interval or on scene change, sends each frame to a vision model to pull out all visible text, and combines results with a Whisper audio transcript.

### Usage

```bash
# Both audio transcript + frame OCR (default, one frame every 5s)
python video_extractor.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Wistia video
python video_extractor.py --url "https://fast.wistia.com/embed/iframe/VIDEO_ID"

# Wistia via page embed URL
python video_extractor.py --url "https://example.com/page?wvideo=VIDEO_ID"

# Scene-change sampling — only capture when screen content changes
python video_extractor.py --url "URL" --scene-threshold 25

# Coarser interval (one frame every 10s)
python video_extractor.py --url "URL" --frame-interval 10

# OpenAI GPT-4o vision instead of Ollama
python video_extractor.py --url "URL" --backend openai

# Audio transcription only (skip vision OCR)
python video_extractor.py --url "URL" --no-frames

# Frame OCR only (skip audio)
python video_extractor.py --url "URL" --no-whisper

# Larger Whisper model for better accuracy
python video_extractor.py --url "URL" --whisper-model small
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--url` | required | Video URL to download and analyze (YouTube, Wistia, or any yt-dlp-supported URL) |
| `--backend` | `ollama` | Vision backend for frame OCR: `ollama` or `openai` |
| `--whisper-model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--no-whisper` | off | Skip audio transcription |
| `--no-frames` | off | Skip frame extraction and vision OCR |
| `--frame-interval` | `5` | Seconds between frames in interval mode |
| `--scene-threshold` | off | Enable scene-change sampling; value is mean pixel diff sensitivity (e.g. `25`). Lower = more frames |
| `--min-scene-gap` | `1.0` | Minimum seconds between saved frames in scene mode |

### Output

Output paths are derived from the video ID (same resolution rules as `product_demo_video_analyzer.py`):

- `artifacts/frames/<video_id>/frames/` — JPEG frames named `frame_<index>_<timestamp>s.jpg`
- `artifacts/transcripts/<video_id>.txt` — Whisper transcript (with timestamps) followed by per-frame vision OCR text
- `artifacts/reports/<video_id>_timeline.html` — visual timeline table: timestamp | frame image | OCR text | audio transcript (open in browser)

### Prerequisites

- Ollama running locally with `llama3.2-vision` pulled (`ollama pull llama3.2-vision`)
- Or an `OPENAI_API_KEY` in a `.env` file for the `openai` backend
- `ffmpeg` installed

---

This project extracts YouTube demo content and answers two questions about every moment in the video:

| Question | Source | Where in the code |
|---|---|---|
| **"What text is visible on screen?"** | Frame OCR via vision model (`--ocr-backend`) | `extract_text_from_frame()` → stored in `window["ocr_text"]` → surfaced in timeline HTML column, macro-chunk prompt, and vision summary prompt |
| **"What is the presenter doing and why?"** | Audio transcript (YouTube API / yt-dlp / Whisper) + Ollama analysis | `get_transcript_segments_*()` → `analyze_transcript_with_ollama()` / `analyze_macro_chunk_window_with_vision()` / `analyze_with_ollama_vision()` |

Both streams feed into every output: the timeline shows them side by side, and Ollama analysis prompts include both.

## Pipeline

```text
[YouTube Video]
    |
    +--> Download Video (yt-dlp)
    |
    +--> Transcript Source (priority order)
    |      1) youtube-transcript-api
    |      2) yt-dlp subtitles
    |      3) Whisper fallback (audio -> text)
    |
    +--> Keyframes  (--keyframe-mode interval | scene)
    |      interval: one frame every --keyframe-seconds (default)
    |      scene:    one frame per distinct screen state (scene-change detection)
    |          |
    |          +--> OCR (--ocr-backend ollama | openai | none)
    |                   extracts visible text from each keyframe
    |
    +--> Macro-chunking: dense 6 fps capture -> 15s windows -> keep 8 diverse frames/window
    +--> Timeline:       keyframe + transcript + OCR per time window -> HTML / Markdown / JSON
    +--> Vision summary: keyframes + transcript + OCR -> single Ollama report
    +--> Analysis:       transcript -> structured PM/UX/dev report
```

## Prerequisites

- Python 3.9+
- Ollama running locally (`http://127.0.0.1:11434`)
- `ffmpeg` installed (for Whisper audio extraction)

Install Python dependencies:

```bash
pip install -U yt-dlp youtube-transcript-api opencv-python openai-whisper
```

## Supported Video Sources

### Input modes

Three ways to provide a video, each changing how the pipeline acquires the transcript:

| Mode | Command | Download | Transcript chain |
|---|---|---|---|
| **URL only** | `--url URL` | yt-dlp downloads automatically | YouTube API → yt-dlp subtitles → Whisper |
| **Local file only** | `--file path.mp4` | Skipped | Whisper only (no YouTube source) |
| **URL + local file (paired)** | `--url URL --file path.mp4` | Skipped (uses local file) | YouTube API → yt-dlp subtitles → Whisper |

Use `--url + --file` together when you have already downloaded the video and don't want to re-download it, but still want to try pulling the YouTube transcript before falling back to Whisper.

When both are used, counts must match — first `--url` pairs with first `--file`, second with second, etc.

```bash
# URL only — download and process
python3 product_demo_video_analyzer.py --url "https://youtube.com/watch?v=ID"

# Local file only — Whisper transcript
python3 product_demo_video_analyzer.py --file path/to/demo.mp4

# Pre-downloaded + URL for transcript lookup
python3 product_demo_video_analyzer.py --url "https://youtube.com/watch?v=ID" --file path/to/demo.mp4

# Multiple videos
python3 product_demo_video_analyzer.py --url "URL_1" --url "URL_2"
python3 product_demo_video_analyzer.py --file video1.mp4 --file video2.mp4
```

### Supported platforms

| Platform | `--url` support | Transcript notes |
|---|---|---|
| **YouTube** | ✅ | YouTube API → yt-dlp subtitles → Whisper |
| **Wistia** | ✅ | YouTube API and subtitles always fail — falls through to Whisper automatically |
| **Local file** | N/A (`--file`) | Whisper only |

Wistia is commonly used for product demo and marketing videos (e.g. Procore, Salesforce). Pass the embed URL directly with `--url`; yt-dlp handles the download. Ensure Whisper fallback is enabled (it is by default).

## Step-by-Step

1. Start Ollama and pull models:

```bash
ollama pull llama3.1
ollama pull llama3.2-vision
```

2. Run macro-chunking workflow (recommended for product demos):

```bash
python3 product_demo_video_analyzer.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --macro-chunking \
  --macro-window-seconds 15 \
  --capture-fps 6 \
  --macro-frames-per-window 8 \
  --no-vision-summary \
  --no-analysis \
  --no-timeline
```

3. Review outputs in `artifacts/reports/`.

## Process Details

### 1) Input and Video ID Resolution

A video ID is derived from each input and used to name all output files (`artifacts/reports/<video_id>_*.md`, `artifacts/frames/<video_id>/`, etc.).

The ID is extracted in this priority order:

| URL pattern | Example | ID used |
|---|---|---|
| `youtu.be/<id>` | `youtu.be/ABC123` | `ABC123` |
| `youtube.com/watch?v=<id>` | `youtube.com/watch?v=ABC123` | `ABC123` |
| `fast.wistia.com/embed/iframe/<id>` | `.../iframe/jgbanquqjs` | `jgbanquqjs` |
| Any page with `?wvideo=<id>` | `procore.com/page?wvideo=jgbanquqjs` | `jgbanquqjs` |
| Any URL — last path segment | `vimeo.com/123456789` | `123456789` |
| No useful path (fallback) | `example.com/?token=xyz` | 12-char MD5 hash of the full URL |

The MD5 fallback is a stable, unique identifier — the same URL always produces the same 12 characters, so output filenames are consistent across runs even when no readable ID can be extracted. In practice, almost all video hosting URLs have a usable ID in the path and the hash is never needed.

For `--file` inputs, the filename stem is used as the video ID (e.g. `demo_video.mp4` → `demo_video`).

### 2) Video Download

- The script downloads a capped-resolution MP4 stream with `yt-dlp`.
- Video download is required for frame extraction and Whisper fallback.
- Files are stored in:
  - `artifacts/videos/<video_id>.*`

### 3) Transcript Acquisition (Fallback Chain)

"Subtitles" and "transcript" mean the same thing in this pipeline — they are different *sources* for the same data structure. All three methods produce identical `[{start, end, text}]` timestamped segments. The pipeline uses whichever source succeeds first; everything downstream (timeline, analysis, OCR alignment) sees one unified transcript regardless of where it came from.

| Source | How it works | When it's used |
|---|---|---|
| **YouTube transcript API** | Fetches pre-existing captions directly from YouTube | First attempt; fastest when available |
| **yt-dlp subtitles** | Downloads `.vtt` subtitle file from YouTube and parses it into segments | Fallback if API fails |
| **Whisper** | Extracts audio from the video and transcribes it locally | Final fallback, or always used for local files and Wistia |

The pipeline tries them in this order and stops at the first success:

```
1. youtube-transcript-api   →  [{start, end, text}, ...]
2. yt-dlp .vtt subtitles    →  [{start, end, text}, ...]
3. Whisper (audio → text)   →  [{start, end, text}, ...]
```

For **local files** (`--file`) and **Wistia URLs**, sources 1 and 2 always fail — the pipeline falls straight through to Whisper.

Normalization applied to all sources before use:

- remove bracketed cues (`[Music]`, etc.)
- collapse repeated whitespace
- remove near-duplicate adjacent lines
- keep timestamped segments (`start`, `end`, `text`)

### 4) Whisper Path (when needed)

When Whisper is the transcript source:

- audio is extracted from video via `ffmpeg`:
  - mono channel, `16kHz` sample rate
- Whisper transcribes audio into timestamped segments
- audio files stored in `artifacts/audio/<video_id>.wav`

### 5) Timeline Generation (Inspection Mode)

For `--timeline` mode:

- transcript is grouped into fixed windows (`--timeline-window-seconds`, default `20`)
- one midpoint frame per window is extracted
- side-by-side inspection artifacts are generated:
  - HTML table with time window + frame + transcript
  - markdown and raw JSON

Frame assets are stored in:
- `artifacts/frames/<video_id>/`

### 6) Macro-Chunking (Workflow Mode)

For `--macro-chunking` mode:

- video is sampled at `--capture-fps` (default `6 fps`)
- samples are grouped into fixed windows (`--macro-window-seconds`, default `15s`)
- each window keeps up to `--macro-frames-per-window` frames (default `8`)
- each window is analyzed by the Ollama vision model with its transcript snippet
- all window analyses are then compiled into one final summary

This is tuned for product demos where a 15-second block typically captures one coherent workflow step.

Frame assets are stored in:
- `artifacts/frames/<video_id>/macro_chunks/`

### 7) Vision Summary (Sparse Keyframe Mode)

For `--vision-summary` mode:

- one keyframe is extracted every `--keyframe-seconds`
- up to `--max-vision-frames` are sent to the vision model
- transcript text is included in the same prompt
- output is a single full report:
  - `artifacts/reports/<video_id>_vision_summary.md`

### 8) Structured PM/UX/Dev Analysis (Text Model)

For `--analysis` mode:

- full transcript text is sent to a text model
- output includes:
  - summary
  - key entities
  - workflow stages
  - PM view
  - UX view
  - developer view
  - open questions

## Common Run Modes

Use these as starting points. Mix and match flags to suit your needs.

---

### A) Timeline only — fastest, no Ollama needed

Inspect transcript and frames side by side. No AI analysis. Works even if Ollama is not running.

```bash
python3 product_demo_video_analyzer.py \
  --url "URL" \
  --no-analysis \
  --no-vision-summary \
  --no-macro-chunking \
  --timeline-window-seconds 15
```

Output: `_timeline.html`, `_timeline.md`, `_timeline.json`

---

### B) Timeline + OCR — read on-screen text, no Ollama needed

Adds a 4th column to the timeline showing visible UI text per frame. Uses vision model for OCR only (one image at a time, much lighter than analysis).

```bash
python3 product_demo_video_analyzer.py \
  --url "URL" \
  --no-analysis \
  --no-vision-summary \
  --no-macro-chunking \
  --ocr-backend ollama \
  --keyframe-mode scene
```

Output: `_timeline.html` with OCR column

---

### C) Transcript-only PM/UX/Dev analysis — no frames needed

Answers "what is the presenter doing and why?" from audio alone. Fast, low memory.

```bash
python3 product_demo_video_analyzer.py \
  --url "URL" \
  --analysis \
  --no-vision-summary \
  --no-timeline \
  --no-macro-chunking
```

Output: `_analysis.md`

---

### D) Vision summary — single report from keyframes + transcript

One Ollama call per keyframe batch. Lighter than macro-chunking. Good when Ollama is slow.

```bash
python3 product_demo_video_analyzer.py \
  --url "URL" \
  --vision-summary \
  --no-macro-chunking \
  --no-analysis \
  --keyframe-seconds 15 \
  --max-vision-frames 12
```

Output: `_vision_summary.md`

---

### E) Vision summary + OCR — both questions answered, without macro-chunking

Recommended when Ollama is slow or you want to avoid macro-chunk timeouts.

```bash
python3 product_demo_video_analyzer.py \
  --url "URL" \
  --vision-summary \
  --no-macro-chunking \
  --no-analysis \
  --ocr-backend ollama \
  --keyframe-seconds 15 \
  --max-vision-frames 12
```

Output: `_vision_summary.md` (includes OCR text), `_timeline.html` with OCR column

---

### F) Macro-chunk workflow analysis — best for step-by-step demo breakdown

Dense frame capture in 15s windows, each analyzed by Ollama. Most detailed but heaviest on memory. Use reduced fps/frames if Ollama is slow.

```bash
python3 product_demo_video_analyzer.py \
  --url "URL" \
  --macro-chunking \
  --no-vision-summary \
  --no-analysis \
  --no-timeline \
  --capture-fps 3 \
  --macro-frames-per-window 4
```

Output: `_macro_chunk_analysis.md`, `_macro_chunk_summary.md`

---

### G) Full pipeline — everything

Timeline + macro-chunks + vision summary + PM/UX/dev analysis + OCR. Runs everything; expect a long runtime.

```bash
python3 product_demo_video_analyzer.py \
  --url "URL" \
  --ocr-backend ollama \
  --capture-fps 3 \
  --macro-frames-per-window 4 \
  --keyframe-seconds 15 \
  --max-vision-frames 12
```

Output: all report files under `artifacts/reports/<video_id>_*`

---

### H) Wistia video — always uses Whisper for transcript

YouTube transcript API and subtitles will fail for Wistia. Pipeline falls through to Whisper automatically; just ensure `ffmpeg` is installed.

```bash
python3 product_demo_video_analyzer.py \
  --url "https://fast.wistia.com/embed/iframe/VIDEO_ID" \
  --no-analysis \
  --no-macro-chunking \
  --ocr-backend ollama \
  --keyframe-mode scene
```

---

### I) Multiple videos in one run

```bash
python3 product_demo_video_analyzer.py \
  --url "URL_1" \
  --url "URL_2" \
  --no-macro-chunking \
  --vision-summary
```

## Macro-Chunk Outputs

When `--macro-chunking` is enabled:

- `artifacts/reports/<video_id>_macro_chunk_analysis.md`
  - Per-window analysis with transcript snippet and selected frames
- `artifacts/reports/<video_id>_macro_chunk_summary.md`
  - Consolidated end-to-end summary across all windows

## Keyframe Modes

Keyframes are the frames extracted from the video for the timeline, vision summary, and OCR. Two modes are available:

### Interval mode (default)

One frame every `--keyframe-seconds` (default: 20). Timestamps are evenly spaced and align predictably with the transcript windows.

```bash
python3 product_demo_video_analyzer.py --url "..." --keyframe-seconds 10
```

### Scene-change mode

A frame is saved only when the screen visually changes — measured as the mean pixel difference between the current frame and the last saved one. If the screen stays static for 30 seconds you get one frame; if it transitions rapidly you get many. This mode produces exactly one frame per distinct UI state, which is ideal for OCR since you're not wasting frames on unchanged content or missing a brief but important screen change.

```bash
# Save a frame whenever the screen changes by more than 25 mean pixel units
python3 product_demo_video_analyzer.py --url "..." --keyframe-mode scene

# More sensitive (more frames, catches subtle changes)
python3 product_demo_video_analyzer.py --url "..." --keyframe-mode scene --scene-threshold 10

# Less sensitive (fewer frames, only major transitions)
python3 product_demo_video_analyzer.py --url "..." --keyframe-mode scene --scene-threshold 40

# Enforce at least 2 seconds between saves
python3 product_demo_video_analyzer.py --url "..." --keyframe-mode scene --min-scene-gap 2
```

**Trade-off**: Scene-change frames have uneven timestamps, so transcript alignment per window is slightly less precise than interval mode. For OCR-focused runs this is usually worth it; for transcript-heavy analysis interval mode works better.

## OCR (On-Screen Text Extraction)

Add `--ocr-backend` to extract visible UI text from keyframes. OCR output is added as a fourth column in the timeline HTML and included in all analysis prompts alongside the audio transcript.

```bash
# OCR with local Ollama (llama3.2-vision)
python3 product_demo_video_analyzer.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --ocr-backend ollama

# OCR with OpenAI GPT-4o (requires OPENAI_API_KEY in .env)
python3 product_demo_video_analyzer.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --ocr-backend openai

# OCR only — no Ollama analysis
python3 product_demo_video_analyzer.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --ocr-backend ollama \
  --no-analysis \
  --no-vision-summary \
  --no-macro-chunking
```

When OCR is enabled, the timeline HTML gains a fourth column **On-Screen Text (OCR)** showing text visible in the video frame (UI labels, menus, data values, headers).

## Whisper Fallback

Whisper is used only if YouTube transcript sources fail (unless disabled).

- Disable fallback:

```bash
--no-whisper-fallback
```

- Change Whisper model:

```bash
--whisper-model small
```

## Troubleshooting

- `Could not reach Ollama...`: start Ollama app/daemon.
- `Missing dependency: yt_dlp` or similar: install packages listed above.
- No transcript found: keep Whisper fallback enabled and ensure `ffmpeg` is installed.

### Ollama Vision Slow or Timing Out

When `llama3.2-vision` is memory-constrained or slow, vision requests may time out. The pipeline handles this automatically in two layers:

**Adaptive image reduction** — if a vision request fails (timeout, OOM, HTTP 500), the pipeline retries the same window with half as many images. It reduces until it reaches 1 image, then falls back.

**Transcript-only fallback** — if all image counts fail, the window output is replaced with its transcript text. The run continues; no data is lost.

You will see messages like:
```
[ollama] Vision request failed; retrying with fewer images (6 -> 3).
[ollama] Vision request failed; retrying with fewer images (3 -> 1).
[sv8uzX66gxw] Vision chunk failed (...): Ollama request timed out after 120s.
[sv8uzX66gxw] Falling back to transcript-only chunk analysis for this window.
```

Each attempt times out in at most 120 seconds (scaled down further for smaller image counts), so the worst case per window is about 3 minutes before moving on.

**To reduce Ollama load:**

```bash
# Fewer frames per window (default 8 → 4)
--macro-frames-per-window 4

# Lower capture rate (default 6 fps → 2)
--capture-fps 2

# Skip macro-chunking entirely; use lighter vision summary instead
--no-macro-chunking --vision-summary --keyframe-seconds 15 --max-vision-frames 12

# Skip all vision analysis; use OCR for on-screen text instead
--no-macro-chunking --no-vision-summary --ocr-backend ollama
```

## Disk Usage Notes

With `--frame-width 960`, you usually will not exceed disk for a few videos, but large batches can grow quickly.

Typical JPEG size at width 960 is often around `100-300 KB` per frame (depends on UI complexity and motion).

Approximate storage per hour of source video:

- Timeline mode (`1 frame / 20s`): ~`180` frames/hour -> about `18-54 MB/hour`
- Macro-chunk mode (`8 frames / 15s` window): ~`1920` frames/hour -> about `190-575 MB/hour`
- Downloaded source video in `artifacts/videos/` can add hundreds of MB per hour

Main disk contributors:

1. `artifacts/frames/`
2. `artifacts/videos/`
3. `artifacts/audio/` (when Whisper fallback is used)

To reduce disk usage:

- use `--frame-width 640`
- use `--macro-frames-per-window 4`
- use `--capture-fps 3`
- use `--keyframe-seconds 30`
- periodically delete old `artifacts/frames`, `artifacts/videos`, and `artifacts/audio`

## CLI Reference

| Option | Default | Description |
|---|---|---|
| `-h`, `--help` | None | Show help text and exit. |
| `--url URL` | — | Video URL (YouTube or Wistia). Repeat for multiple. At least one of `--url` or `--file` required. |
| `--file PATH` | — | Local video file path. Use alone (Whisper only) or pair 1:1 with `--url` (skips download, URL still used for transcript lookup). |
| `--model MODEL` | `llama3.1` | Text model for transcript-based PM/UX/dev analysis and macro-chunk final summary. |
| `--vision-model VISION_MODEL` | `llama3.2-vision` | Vision model for image+transcript analysis. |
| `--work-dir WORK_DIR` | `artifacts` | Base output directory for generated files. |
| `--download-video` | `False` | Keep downloaded files in `artifacts/videos/`. |
| `--analysis` / `--no-analysis` | Enabled | Enable/disable structured transcript-only PM/UX/dev report output (`--no-analysis` skips `<video_id>_analysis.md`). |
| `--vision-summary` / `--no-vision-summary` | Enabled | Enable/disable sparse keyframe multimodal summary. |
| `--macro-chunking` / `--no-macro-chunking` | Enabled | Enable/disable windowed macro-chunking workflow (`--no-macro-chunking` skips `<video_id>_macro_chunk_analysis.md` and `<video_id>_macro_chunk_summary.md`). |
| `--timeline` / `--no-timeline` | Enabled | Enable/disable timeline outputs (frame + transcript). |
| `--timeline-window-seconds N` | `20` | Time window size for timeline generation. |
| `--keyframe-seconds N` | `20` | In vision-summary mode, extract one keyframe every `N` seconds. |
| `--macro-window-seconds N` | `15` | Window size (seconds) for macro-chunking. |
| `--capture-fps N` | `6` | Frame sampling rate for macro-chunking. |
| `--macro-frames-per-window N` | `8` | Max representative frames kept/sent per macro window. |
| `--frame-width N` | `960` | Resize extracted frames to this max width. |
| `--max-vision-frames N` | `24` | Max total frames sent in sparse keyframe vision-summary mode. |
| `--whisper-fallback` / `--no-whisper-fallback` | Enabled | Enable/disable Whisper fallback if transcript API/subtitles fail. |
| `--whisper-model MODEL` | `base` | Whisper model used during fallback transcription. |
| `--ocr-backend` | `none` | Vision backend for per-frame OCR text extraction: `ollama`, `openai`, or `none`. When enabled, visible on-screen text is extracted from each keyframe and included in the timeline, vision summary, and macro-chunk analysis. |
| `--keyframe-mode` | `interval` | Keyframe extraction mode: `interval` (one frame every `--keyframe-seconds`) or `scene` (one frame per distinct screen state, driven by pixel difference). |
| `--scene-threshold` | `25` | Mean pixel difference threshold for `--keyframe-mode scene`. Lower = more sensitive, more frames. Range 0–255. |
| `--min-scene-gap` | `1.0` | Minimum seconds between saved frames in scene-change mode. |
