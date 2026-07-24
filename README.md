# VideoExtractor: YouTube Demo Analysis Pipeline

This project contains three tools that answer different questions about a product demo video:

| Script | Question answered | Backend | Best for |
|---|---|---|---|
| `video_extractor.py` | **"What text is visible on screen?"** | Ollama / OpenAI | Reading UI elements, labels, menus, and data values shown in the demo |
| `product_demo_video_analyzer.py` | **"What is the presenter doing and why?"** | Ollama (`qwen2.5vl:32b`) | Laptop pipeline. Same Qwen2.5-VL model family as the DGX build, so Mac runs preview DGX output. |
| `product_demo_video_analyzer_dgx.py` | **"What is the presenter doing and why?"** | vLLM (`Qwen2.5-VL-32B-AWQ`) + RapidOCR + pHash-gated keyframing | Same questions, rebuilt for NVIDIA GB10 Grace-Blackwell hardware (MSI EdgeXpert / DGX Spark). ~10-25× faster than the Ollama pipeline on the same box. |

If you're on an NVIDIA GB10 Blackwell box (EdgeXpert / DGX Spark), use `_analyzer_dgx.py`. If you're on a laptop with Ollama, use the original `_analyzer.py`. The rest of this README explains both.

---

## Why `product_demo_video_analyzer_dgx.py` exists — a hardware-driven rewrite

The original `product_demo_video_analyzer.py` runs Ollama-hosted `qwen2.5vl:32b` over HTTP. On a laptop this is **too slow to be practical**: a single vision summary over 20 keyframes takes 5-10 minutes, and macro-chunking (~80 vision calls on a 20-min video) routinely times out or OOMs. Ollama's REST loop can't batch multi-image prompts and the local model quantization doesn't touch any modern accelerator kernels.

`_analyzer_dgx.py` rebuilds the same pipeline for the **NVIDIA GB10 Grace-Blackwell platform** (MSI EdgeXpert, NVIDIA DGX Spark, and other GB10-partner boxes: 20-core Grace ARM CPU + Blackwell GPU + 128 GB unified LPDDR5X). Three architectural decisions drive the whole rewrite:

### 1. Inference engine: vLLM instead of Ollama

Ollama uses **llama.cpp** under the hood. On any platform, llama.cpp dequantizes 4-bit weights to bf16 for every matmul — Blackwell's native FP4 tensor cores are never touched. The HTTP REST loop also serialises image encoding on the client side, so multi-image prompts are sent sequentially rather than batched.

**vLLM's `LLM.chat()` API prefills all images in one pass** and dispatches to Blackwell tensor-core kernels directly (AWQ-Marlin on AWQ checkpoints). The engine swap alone gives 3-5× throughput on GB10 before quantization changes are factored in.

Where Ollama sits on your GB10 box (32B model, 24 keyframes, 20-min video):

| Backend | Vision summary wall-clock | Notes |
|---|---|---|
| **vLLM + AWQ-Marlin** | **~4 min** | Default; Marlin kernel auto-selected by vLLM |
| vLLM + NVFP4 | ~2-3 min | Requires pre-quantized NVFP4 checkpoint + `--quantization nvfp4` |
| Ollama + Qwen2.5-VL-32B | ~8-12 min | llama.cpp / bf16-dequant path; ~30-50% of vLLM throughput |
| Ollama on laptop (e.g., M2) | ~30-40 min | Often fails mid-run with vision timeouts |

If your instinct is "just use Ollama — it's easier to set up", note that setup pain is a one-time cost and Ollama on GB10 is roughly 2-3× *slower* than vLLM on the same box because it hits the same bf16-dequant path bitsandbytes NF4 did.

### 2. Model: Qwen2.5-VL-32B-AWQ, not 72B

Blackwell's on-package unified memory (128 GB LPDDR5X, ~273 GB/s bandwidth) is generous for weights but modest on bandwidth compared to HBM3. Vision transformers are memory-bandwidth-bound at decode, so **model size trades off nearly linearly against throughput**:

| Model | Weights (AWQ) | Prefill (24 images) | Decode (2048 tokens) | Vision summary end-to-end |
|---|---|---|---|---|
| Qwen2.5-VL-72B-AWQ | ~40 GB | ~90 s | ~600 s (3.4 tok/s) | ~11 min |
| **Qwen2.5-VL-32B-AWQ** (default) | ~17 GB | ~35 s | ~180 s (11 tok/s) | **~4 min** |
| Qwen2.5-VL-7B-AWQ | ~5 GB | ~10 s | ~50 s (40 tok/s) | ~1 min |

32B is the practical sweet spot — quality is indistinguishable from 72B for structured demo-summarization prompts, and wall-clock is ~3× faster. Swap to 7B via `--vision-model Qwen/Qwen2.5-VL-7B-Instruct-AWQ` when you want interactive iteration.

### 3. Quantization: AWQ-Marlin now, NVFP4 later

Blackwell's headline "1 PFLOP FP4" throughput requires **native FP4 tensor-core kernels** — NVFP4 or MXFP4 formats with weights pre-quantized offline and served through TensorRT-LLM or vLLM's FP4 backend. The previous plan-of-record (`bitsandbytes` NF4) is a *storage* trick: weights are packed to 4 bits on disk but **dequantized to bf16 on every matmul**, so compute stays on the older bf16 tensor cores. You get the memory savings but none of Blackwell's FP4 hardware advantage.

The current default (`--quantization` unset → auto-detects AWQ from the checkpoint → picks the `awq_marlin` kernel on Blackwell) gets ~3× the throughput of bf16 at the same VRAM footprint as bitsandbytes NF4, with no rebuild step. Users with an NVFP4-quantized checkpoint can pass `--quantization nvfp4` and swap the model ID to unlock the full FP4 path — no other code changes.

### How the precision formats actually work — NF4, BF16, AWQ, NVFP4

Understanding why NF4 is "fake 4-bit" requires a quick look at how floating-point formats trade bits between *range* and *precision*, and what "block scaling" changes about that trade-off.

#### BF16 — the baseline compute format

A floating-point number is represented as:
```
value = (-1)^sign × 2^(exponent − bias) × (1 + mantissa / 2^M)
```

| Format | Total bits | Sign | Exponent | Mantissa | Dynamic range | Precision steps |
|---|---|---|---|---|---|---|
| FP32 | 32 | 1 | 8 | 23 | ±3.4 × 10^38 | ~16M per exponent step |
| **BF16** | 16 | 1 | **8** | 7 | ±3.4 × 10^38 (same as FP32) | ~128 per exponent step |
| FP16 | 16 | 1 | 5 | 10 | ±65504 (much smaller) | ~1024 per exponent step |

BF16 keeps FP32's 8-exponent-bit dynamic range but cuts mantissa precision to 7 bits. This is a deliberate ML trade-off: neural network weights span many orders of magnitude (need wide range) but don't require fine-grained precision (coarse steps are fine). **BF16 tensor cores are the standard compute path on Ampere/Hopper; Blackwell has them too, but adds faster FP4 units on top.**

#### NF4 — storage-only "4-bit", not compute 4-bit

bitsandbytes `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` stores weights as NormalFloat 4 (NF4). Here is what NF4 actually does:

1. **Offline**: divide each 64-weight block by its absmax to normalize to [-1, 1]. Map each value to the nearest of 16 quantization levels. The 16 levels are NOT uniformly spaced — they are the 16 *quantiles* of a standard normal distribution:
   ```
   {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0,
     0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626, 0.7229, 1.0}
   ```
   More levels cluster near zero because neural network weights follow an approximately normal distribution — most weights are small, few are large. NF4 is information-theoretically optimal for this distribution.

2. **At inference (every single matmul)**: bitsandbytes calls `dequantize_blockwise()`, reconstructing BF16 tensors from the stored 4-bit indices. The actual `nn.Linear` call then runs on **BF16 CUDA cores**, not FP4 hardware.

**The consequence**: you get the memory saving (4× smaller than BF16) but the computation cost is the same as BF16 plus a dequantize overhead. Blackwell's FP4 tensor-core units sit idle.

#### AWQ — activation-aware INT4, with fused Marlin kernels

AWQ (Activation-Aware Weight Quantization) also stores weights in 4 bits, but differently:

1. **Observation**: quantization error matters most for weights that multiply *large activations*. A 1% error on a weight paired with an activation of magnitude 100 does 100× more damage than the same error on a weight paired with an activation of magnitude 1.
2. **AWQ offline pass**: find a per-channel scale `s` that minimises `‖(W/s) − round(W/s)‖` weighted by activation magnitudes. This scale is absorbed into the adjacent normalization layer — no runtime overhead.
3. **Result**: INT4 weights packed in a group-wise layout that Marlin kernels can consume.

**Marlin (the kernel vLLM uses on Blackwell)**: fuses the INT4→BF16 dequantize step and the matrix multiply into a single CUDA kernel that saturates memory bandwidth. It reads 4-bit weights, reconstructs BF16 on-the-fly in shared memory, and dispatches to BF16 tensor cores — all without the separate `dequantize_blockwise()` call that bitsandbytes needs. The fused kernel is ~3× faster than bitsandbytes NF4 at the same weight size because:
- No separate dequantize kernel launch
- BF16 tensor cores see a continuous stream of work
- Memory reads are 4× smaller, so LPDDR5X bandwidth goes 4× further

AWQ-Marlin is still using **BF16 tensor cores**, not FP4. It just reaches them much more efficiently.

#### NVFP4 / MXFP4 — real 4-bit compute via block scaling

FP4 (E2M1 format: 1 sign, 2 exponent, 1 mantissa) has only 16 representable values and a tiny range — far too coarse for isolated weights. The trick that makes it usable is **microscaling**:

```
weight_block = [w₁, w₂, ..., w₃₂]   ← 32 weights share one FP8 scale
actual_value[i] = w_fp4[i] × scale_fp8
```

1. Every 32 weights in a column get one shared FP8 scale factor (stored separately, 8 bits).
2. Each weight is then stored as FP4 relative to that scale.
3. The tensor core performs: `output = (W_fp4 × scale_fp8) @ activation` **entirely in FP4 arithmetic** — no dequantization to BF16 first.

Why does this preserve accuracy despite only 1 mantissa bit in FP4?
- The FP8 scale covers the block's dynamic range (8 exponent bits → same range as BF16)
- FP4 encodes the *relative* precision within the block (think of it as: the scale is the "big ruler", FP4 is the "tick marks")
- Neighbouring weights in the same column tend to have similar magnitudes (smooth weight landscapes), so one shared scale loses little information

**The result**: the matmul truly runs through Blackwell's dedicated FP4 tensor cores. No intermediate BF16. This is where the "1 PFLOP FP4" figure comes from — FP4 tensor cores deliver 2× the FLOPs of BF16 tensor cores at the same silicon area.

#### Side-by-side comparison

| Format | Bits/weight | Compute path | Dequantize to BF16? | Tensor cores used | Throughput on GB10 (32B) |
|---|---|---|---|---|---|
| BF16 (no quant) | 16 | Direct matmul | No | BF16 | 1× (baseline) |
| bitsandbytes NF4 | 4 (storage only) | Dequantize → BF16 matmul | **Yes, every call** | BF16 | ~0.9× (overhead) |
| **AWQ-Marlin** (default) | 4 (INT4 + scale) | Fused dequantize + matmul | Fused, in shared mem | BF16 | **~3×** |
| NVFP4 / MXFP4 | 4 (true compute) | Native FP4 matmul | **No** | FP4 | ~6× (projected) |

**Bottom line**: NF4 (bitsandbytes) gives memory savings but no compute speed-up on Blackwell. AWQ-Marlin is the practical maximum today. NVFP4 is the ceiling but requires a pre-quantized checkpoint and vLLM FP4 backend support — not yet mainstream for Qwen2.5-VL.

## OCR swap: PaddleOCR → RapidOCR

The original plan targeted PaddleOCR with TensorRT, but on aarch64 + CUDA 12.8:

- `paddlepaddle-gpu` wheels lag the CUDA version and often silently fall back to CPU.
- PaddleOCR 2.x → 3.x removed the `use_gpu`, `use_tensorrt`, `use_angle_cls`, and `enable_mkldnn` kwargs. Code written against the old API breaks on install.
- The `enable_mkldnn` path is CPU-only Intel-DNN — meaningless on Grace ARM anyway.

**RapidOCR (ONNX Runtime)** ships clean ARM-native wheels, has a stable API across versions, and doesn't need CUDA at all. The catch — CPU by default — is compensated by the pHash prefilter below.

## Keyframing: dHash prefilter + OCR-gated saves

The `--keyframe-mode ocr` path in the original `_analyzer.py` would run PaddleOCR at 1 FPS across the entire video — ~1800 OCR calls on a 30-min video, just to detect frames where the on-screen text changed. Even fast OCR at that volume dominates wall-clock time.

`_analyzer_dgx.py` gates OCR behind a **dHash (perceptual hash) prefilter**:

1. Compute an 8×8 dHash of each 1-FPS sample (a few microseconds per frame).
2. If Hamming distance from the last saved frame ≤ `--phash-threshold` (default `5/64` bits ≈ 8% visual change), **skip OCR entirely** — the frame is visually similar to what's already saved.
3. Only run OCR on frames that pass the visual-change gate.
4. Save the frame if OCR text similarity < `--ocr-similarity-threshold` (default 0.85).

Numbers from a real 20-minute construction demo run on EdgeXpert:

```
[ocr-filter] Done: 56 keyframes saved, 80 OCR calls, 58 phash skips.
```

That's 80 OCR calls versus the ~1200 the naive approach would have made — **~93% reduction in OCR work**. CPU RapidOCR handles the residual 80 calls in <10 seconds total on Grace.

## Macro-chunking: now off by default in `_analyzer_dgx.py`

Macro-chunking was the original heavyweight path in `_analyzer.py`: dense 6-FPS sampling into 15-second windows, one LLM vision call per window (~80 calls on a 20-min video). It existed to compensate for **interval-based keyframing** missing UI state transitions between samples.

With OCR-gated keyframes, that redundancy disappears — keyframes are already state-aware, one per distinct on-screen state. The single `analyze_with_qwen_vision` call over 24 sampled keyframes captures the same content as macro-chunking's 80 calls, at ~25× less compute:

| Pipeline | LLM vision calls | Wall time on GB10 (20-min video, 32B AWQ) |
|---|---|---|
| Vision summary (24 keyframes, 1 call) | 1 | **~1 min** |
| Macro-chunking (80 windows × 8 frames) | ~80 | ~25-40 min |

`_analyzer_dgx.py` defaults `--macro-chunking` to **off**. Turn it on for edge cases: long-form videos (45+ min) where 24 keyframes can't hold all state, purely-visual content (chart animations, 3D rotations) that OCR misses, or downstream tools that need per-timestamp analysis.

## Recommended run on GB10 hardware

```bash
python3 product_demo_video_analyzer_dgx.py \
  --url "https://youtu.be/..." \
  --keyframe-mode ocr \
  --phash-threshold 5
```

That's it. `--analysis`, `--vision-summary`, and `--timeline` are on by default; `--macro-chunking` is off. Total wall time on a 20-min demo: **~5-8 min end-to-end** (transcript + keyframes + vision summary + PM/UX/dev analysis). The equivalent run on `_analyzer.py` with Ollama on a laptop is 45+ min, and often fails partway through with vision timeouts.

## Running multiple videos

`--url` and `--file` both accept `action="append"` — repeat the flag once per video. **vLLM loads the model once** and is reused across every input in the batch, so per-video overhead is transcript + keyframes + LLM calls only (no reload penalty).

**Multiple YouTube/Wistia URLs:**

```bash
python3 product_demo_video_analyzer_dgx.py \
  --url "https://youtu.be/VIDEO_ID_1" \
  --url "https://youtu.be/VIDEO_ID_2" \
  --url "https://youtu.be/VIDEO_ID_3" \
  --keyframe-mode ocr
```

Each video gets its own output directory under `artifacts/reports/<video_id>/` and `artifacts/frames/<video_id>/`.

**Multiple local files:**

```bash
python3 product_demo_video_analyzer_dgx.py \
  --file artifacts/demo_v1.mp4 \
  --file artifacts/demo_v2.mp4 \
  --keyframe-mode ocr
```

**Pairing a URL with a local file (use local copy for video, URL for transcript API):**

Pass `--url` and `--file` in matched order. The n-th `--url` is paired with the n-th `--file`. The URL is used to fetch a YouTube transcript (faster than Whisper); the local `.mp4` is decoded for frames and Whisper fallback — no re-download.

```bash
python3 product_demo_video_analyzer_dgx.py \
  --url "https://youtu.be/VIDEO_ID_1" --file artifacts/demo_v1.mp4 \
  --url "https://youtu.be/VIDEO_ID_2" --file artifacts/demo_v2.mp4
```

**Shell loop for a whole folder** (useful when you have many local files and no remote URL):

```bash
for f in artifacts/*.mp4; do
  python3 product_demo_video_analyzer_dgx.py \
    --file "$f" \
    --keyframe-mode ocr \
    --no-macro-chunking
done
```

The shell loop runs videos sequentially, reloading vLLM per invocation — use the multi-`--file` form above when you want the single-load benefit across all videos.

---

## Pipeline — `product_demo_video_analyzer_dgx.py`

```text
[Input: YouTube URL | Wistia URL | Local .mp4]
    |
    +--> Download (yt-dlp)                [skipped when --file is provided]
    |
    +--> Transcript chain (first success wins)
    |      1) youtube-transcript-api      [YouTube URLs only]
    |      2) yt-dlp .vtt subtitles       [YouTube URLs only]
    |      3) Whisper local transcription [always available — used for Wistia + local files]
    |
    +--> Keyframe extraction  — pick ONE mode via --keyframe-mode
    |      |
    |      +-- interval  [--keyframe-mode interval]
    |      |     Sample one frame every --keyframe-seconds (default 20 s).
    |      |     Fixed cadence, content-agnostic.
    |      |     Best for: predictable transcript alignment; talking-head videos
    |      |     with little UI change.
    |      |
    |      +-- scene     [--keyframe-mode scene]
    |      |     Save a frame when cv2.absdiff(frame, prev_frame).mean() >= --scene-threshold.
    |      |     Triggers on any visual pixel change; no OCR involved.
    |      |     --min-keyframe-gap enforces a minimum seconds-between-saves floor.
    |      |     Best for: chart animations, 3D rotations, drawings — content that
    |      |     changes visually but not textually.
    |      |
    |      +-- ocr       [--keyframe-mode ocr]   [DEFAULT]
    |            dHash prefilter + RapidOCR text-change detection.
    |            For each 1-FPS sample:
    |              1. dHash(frame) — 64-bit visual fingerprint
    |              2. Hamming distance <= --phash-threshold?  yes: skip (no OCR call)
    |              3. RapidOCR (ONNX Runtime, CPU) reads on-screen text
    |              4. text similarity < --ocr-similarity-threshold?
    |                    yes: save frame + OCR text as keyframe
    |                    no : skip
    |            Best for: UI-heavy product demos where each screen state has
    |            distinct on-screen text (menus, forms, tables).
    |
    +--> Timeline output           HTML / Markdown / JSON side-by-side viewer
    |                               keyframe + transcript window + OCR text per row
    |                               artifacts/reports/<video_id>_timeline.{html,md,json}
    |
    +--> Vision summary            vLLM (Qwen2.5-VL-32B-AWQ, awq_marlin on Blackwell)
    |                               up to --max-vision-frames keyframes + transcript + OCR
    |                               -> single 7-section markdown report
    |                               artifacts/reports/<video_id>_vision_summary.md
    |
    +--> PM / UX / Dev analysis    vLLM (same engine, text-only — no images)
    |                               full transcript (up to 45 K chars)
    |                               -> Summary + Industry Pain Points + Entities + Workflow +
    |                                  PM view + UX view + Dev view + Open Questions +
    |                                  Automation Opportunities
    |                               artifacts/reports/<video_id>_analysis.md
    |
    +--> [OPT-IN] Macro-chunking   OFF by default. Enable with --macro-chunking for
                                   long-form videos (45+ min) or purely-visual content.
                                   6 fps capture -> 15 s windows -> 8 diverse frames/window
                                   -> one vLLM vision call per window (~80 calls / 20 min video)
                                   -> per-window analysis + compiled full summary
                                   artifacts/reports/<video_id>_macro_chunk_{analysis,summary}.md
```

Notes on the flow above:

- **Keyframes are extracted once**, then reused by Timeline and Vision summary — no redundant frame decoding.
- **The vision summary and PM/UX/Dev analysis share the same vLLM engine**, loaded once at process start. The engine stays resident until the run finishes; images are prefilled in a single batch per call.
- **OCR text captured during keyframe extraction is threaded into every downstream prompt** (timeline HTML column, vision summary, macro-chunk analysis) — the model gets both the pixels *and* the exact on-screen text, which dramatically improves accuracy on UI-dense frames.
- **Everything except the LLM calls is CPU-bound** on Grace (RapidOCR, dHash, ffmpeg, Whisper base) — meaning the pipeline saturates the CPU cores while the GPU idles between vision calls. Suitable for concurrent multi-video runs.

---

## Installation on GB10 hardware (DGX Spark / MSI EdgeXpert)

The GB10 platform is aarch64 (Grace ARM) + Blackwell GPU + CUDA 12.8 or CUDA 13 (DGX Spark units shipped from mid-2025 ship CUDA 13). Wheel selection matters — stock PyPI torch has no `sm_121` kernels, and Python 3.14 aarch64 wheels are sparse for many ML libraries.

**The key trick**: install vLLM nightly *first*, before torch. vLLM nightly bundles the correct torch version and all `nvidia-*-cu13` runtime libraries as pip dependencies, so manually installing torch first almost always creates a version conflict. Follow these steps in order.

### Prerequisites

- Ubuntu 24.04 aarch64 (`uname -m` → `aarch64`)
- NVIDIA driver ≥ 570.x (Blackwell requires 570+)
- CUDA 12.8+ runtime (`nvidia-smi` shows CUDA Version at top-right)

**Python 3.12 *with dev headers* — required, not optional.** Stock DGX OS typically ships `python3.12` and `python3.12-venv` but **not** `python3.12-dev`. Without the `-dev` package, Triton's runtime JIT compile fails during vLLM model load (missing `Python.h`, gcc exits 1) — even when everything else is set up correctly. Install all three regardless of what `python3.12 --version` reports:

```bash
# If Python 3.12 isn't installed at all (only 3.14 shipped), add the PPA first:
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update

# Always run — installs Python 3.12 if missing AND guarantees the dev headers are present:
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

Verify the dev headers landed:

```bash
ls /usr/include/python3.12/Python.h    # must exist, or Triton JIT will fail
```

### Pre-flight sanity check

Run these before creating the venv. If any line fails, fix it before continuing — no Python install step will work if the driver or architecture is wrong.

```bash
# System fundamentals
nvidia-smi                           # Blackwell GPU must appear; CUDA Version ≥ 12.8 in top-right corner
uname -m                             # must print aarch64
python3.12 --version                 # must print 3.12.x

# Is libcuda.so.1 in the linker path?
ldconfig -p | grep libcuda.so        # must show at least one libcuda.so.1 entry
                                     # if empty: driver library missing from ld.so.conf — reboot or run ldconfig

# Python dev headers (required for Triton JIT fallback in vLLM stable; good to have regardless)
ls /usr/include/python3.12/Python.h  # must exist — if not: sudo apt install python3.12-dev

# CUDA toolkit (nvcc is NOT required for vLLM nightly — prebuilt kernels are used)
nvcc --version 2>&1 || echo "no nvcc — OK if using vLLM nightly"
ls /usr/local/ | grep cuda           # shows installed CUDA toolkit directories
```

Expected healthy output:

```
nvidia-smi          → Blackwell listed, "CUDA Version: 13.0" (or 12.8)
uname -m            → aarch64
python3.12          → Python 3.12.3
ldconfig libcuda    → libcuda.so.1 → /usr/lib/aarch64-linux-gnu/libcuda.so.1
Python.h            → /usr/include/python3.12/Python.h
nvcc                → Cuda compilation tools, release 13.0 (or "no nvcc — OK")
```

### Step-by-step install

**0. Confirm Python 3.12 is present**

```bash
python3.12 --version   # should print Python 3.12.x
ls /usr/include/python3.12/Python.h   # must exist — see Prerequisites above
```

If either check fails, re-run the `sudo apt install` line in Prerequisites before continuing.

**1. Create a Python 3.12 venv**

```bash
python3.12 -m venv ~/.venvs/videoextractor
source ~/.venvs/videoextractor/bin/activate
python3 -m pip install --upgrade pip wheel setuptools
```

**2. vLLM nightly — installs FIRST because it pulls the correct torch + CUDA libs**

```bash
pip install -U --pre --extra-index-url https://wheels.vllm.ai/nightly vllm
```

This one command brings in vLLM plus its pinned dependencies — currently `torch==2.11.0`, `triton>=3.6`, `cuda-toolkit-13.x`, `nvidia-cublas-13`, `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, and friends. Do **not** `pip install torch` yourself first — you'll almost certainly land on a torch build that vLLM doesn't pin against, and end up in an uninstall/reinstall loop resolving `vllm requires torch==X.Y.Z, but you have torch A.B.C`.

Why nightly, not stable: vLLM stable falls back to Triton runtime JIT for Blackwell (`sm_121`) kernels and typically fails to compile on stock DGX OS. Nightly ships prebuilt sm_121 kernels and skips JIT entirely.

Verify torch + Blackwell are visible from inside the venv:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0), torch.version.cuda)"
# expect: 2.x.x True (12, 1) 13.0
#                      ↑ sm_121  ↑ CUDA 13.0
# If torch.version.cuda says 12.8, that's also fine — cu128 wheels run on CUDA 13 via forward-compat
```

Also confirm vLLM installed cleanly:

```bash
pip show vllm | grep Version         # should show a date-stamped nightly, e.g. 0.9.x.devYYYYMMDD
```

If `get_device_capability` returns `(12, 0)` or lower, your NVIDIA driver is older than 570 — update DGX OS drivers before continuing. If `is_available()` is `False`, the driver-userspace mismatch is deeper; check that `ldconfig -p | grep libcuda.so` returns a result, then restart the shell and re-activate the venv.

**3. Everything else — pure Python and ONNX wheels, all arm64-clean**

```bash
pip install rapidocr-onnxruntime opencv-python-headless yt-dlp openai-whisper youtube-transcript-api python-dotenv accelerate
```

**4. ffmpeg at OS level (Whisper needs it)**

```bash
sudo apt install -y ffmpeg
```

**5. Pre-fetch the model weights (~17 GB one-time download)**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct-AWQ
```

**6. Import smoke test — no models loaded, confirms all code paths parse**

```bash
cd /path/to/VideoExtractor
python3 -c "from product_demo_video_analyzer_dgx import build_qwen_engine, build_ocr_engine, compute_dhash; print('imports ok')"
```

This loads no weights. If it exits with `imports ok`, the venv is wired up correctly.

**7. Keyframe smoke test — validates OCR + pHash pipeline without loading vLLM**

```bash
python3 product_demo_video_analyzer_dgx.py \
  --file artifacts/<short-clip>.mp4 \
  --no-analysis --no-vision-summary --no-macro-chunking \
  --timeline --keyframe-mode ocr
```

Expect `[ocr-filter] Done: N keyframes saved, M OCR calls, K phash skips.` If N is 20-100 and pHash skips > 0, the gate path is healthy.

**8. Full pipeline test with vLLM online**

```bash
python3 product_demo_video_analyzer_dgx.py \
  --url "https://www.youtube.com/watch?v=<short-demo-id>" \
  --keyframe-mode ocr --vision-summary --analysis --no-macro-chunking
```

Expect ~5-8 min wall time for a 20-min video. If vLLM crashes during model load, see the Triton JIT row in the troubleshooting table below.

**9. Save the working environment for future rebuilds**

```bash
pip freeze > requirements-lock.txt
```

Rebuilding from this lock file is faster than re-resolving nightly deps, but the lock will pin a specific nightly build — refresh it monthly.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` returns `command not found` or fails | NVIDIA driver not installed or PATH not set | On DGX/EdgeXpert this shouldn't happen; run `sudo apt install -y nvidia-utils-570` or reinstall OS drivers. Run `nvidia-smi` before anything else — if it doesn't show the Blackwell GPU and CUDA version, no Python step will work. |
| `nvidia-smi` works but `torch.cuda.is_available()` is `False` AND `ldconfig -p \| grep libcuda.so` is empty | libcuda.so.1 isn't registered with the dynamic linker even though the driver is installed | Run `sudo ldconfig` to rebuild the linker cache. If it stays empty: `find /usr /lib -name "libcuda.so*" 2>/dev/null` — if found, add the containing directory to `/etc/ld.so.conf.d/` and re-run `ldconfig`. |
| `torch.cuda.is_available()` is `False` after step 2 | vLLM nightly pulled torch but the CUDA driver isn't visible from Python (usually a driver-userspace mismatch or the venv shell inherited a stripped `LD_LIBRARY_PATH`) | Confirm `nvidia-smi` works. Then in the venv: `python3 -c "import torch; print(torch.version.cuda)"` — if this prints a version but `is_available()` is False, restart the shell / re-activate the venv. Last resort: uninstall & reinstall vLLM nightly so its pinned nvidia-* libs re-register. |
| `pip's dependency resolver ... vllm X requires torch==Y, but you have torch Z is incompatible` | You installed torch manually before vLLM, and torch is now on the wrong version | Uninstall torch and let vLLM own the pin: `pip uninstall -y torch torchvision torchaudio && pip install -U --pre --extra-index-url https://wheels.vllm.ai/nightly vllm` |
| `pip install vllm>=0.6.4` installs but model load fails with Triton JIT errors | Stable vLLM installed (not nightly); you may also have pre-installed a cu128 torch manually. Stable vLLM + CUDA 13.0 system = JIT compile path that often breaks. | Install nightly (which brings its own torch pin): `pip uninstall -y torch torchvision torchaudio vllm && pip install -U --pre --extra-index-url https://wheels.vllm.ai/nightly vllm`. Do **not** install torch separately — nightly vLLM pulls the correct torch + `nvidia-cu13` libs automatically. |
| `get_device_capability` returns `(12, 0)` or lower | NVIDIA driver older than 570 | Update DGX OS drivers |
| vLLM error: "no kernel image for sm_121" | vLLM version predates Blackwell support | Nightly: `pip install -U --pre --extra-index-url https://wheels.vllm.ai/nightly vllm` |
| vLLM crashes during model load with `subprocess.CalledProcessError: Command '['/usr/bin/gcc', ..., '/tmp/tmpXXX/cuda_utils.c', ...]' returned non-zero exit status 1`, or the exposed gcc error says `Python.h: No such file or directory` | vLLM stable is falling back to Triton runtime JIT because no prebuilt Blackwell kernel is available. Triton's shim compile needs `python3.12-dev` (often missing on stock DGX OS). | 1) Install dev headers: `sudo apt install -y python3.12-dev` and verify `/usr/include/python3.12/Python.h` exists. 2) Switch to vLLM nightly to skip the JIT path entirely (ships prebuilt sm_121 kernels): `pip uninstall -y torch torchvision torchaudio vllm && pip install -U --pre --extra-index-url https://wheels.vllm.ai/nightly vllm`. This is the same install flow as step 2 above and is the definitive fix. |
| `Could not find a version that satisfies the requirement torch` (or vllm/onnxruntime) | Python 3.14 venv — many ML wheels aren't there yet on aarch64 | Use `python3.12 -m venv ...` |
| ONNX Runtime prints `Failed to detect devices under "/sys/class/drm/card0"` | GPU probe on Grace unified memory — DRM sysfs doesn't expose the on-package Blackwell | Benign — RapidOCR runs on CPU by design. Silence with `import onnxruntime as ort; ort.set_default_logger_severity(3)` at the top of the script if noisy. |
| vLLM OOM on model load | 32B AWQ + activations exceed the vLLM alloc | Lower `gpu_memory_utilization=0.85` in `build_qwen_engine()` (product_demo_video_analyzer_dgx.py:46), or drop to `--vision-model Qwen/Qwen2.5-VL-7B-Instruct-AWQ` |
| `ModuleNotFoundError: No module named 'vllm'` (or `rapidocr_onnxruntime`, `cv2`, etc.) | venv not activated in this shell | `source ~/.venvs/videoextractor/bin/activate`, then re-run. To verify: `which python` should point inside `~/.venvs/videoextractor/`. |
| `Missing dependency: yt_dlp` | venv not activated | `source ~/.venvs/videoextractor/bin/activate` |
| `[transcript] No subtitles found` / transcript is empty and run takes much longer than expected | YouTube transcript API returned nothing (private/restricted video, Wistia URL, or local file). Pipeline falls back to Whisper, which needs `ffmpeg`. | Run `ffmpeg -version` — if not found, run `sudo apt install -y ffmpeg`. Whisper fallback is on by default; disable only with `--no-whisper-fallback`. For Wistia and local files, Whisper is always the first (and only) transcript source. |
| `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'` during Whisper transcription | `ffmpeg` not installed at OS level | `sudo apt install -y ffmpeg` (install step 4 above). Whisper calls ffmpeg to decode audio — it must be on the system `PATH`, not inside the venv. |
| First run hangs on "Loading model" | HF download in progress silently | Pre-fetch weights (step 5). Set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads. |
| vLLM prints "Initializing vLLM engine" and then nothing for 2-3 minutes | Normal — first-time CUDA kernel compilation and weight loading for a 17 GB checkpoint | Wait. Do not Ctrl+C. Look for `INFO: Model loaded` (or similar) to confirm it finished. Subsequent runs are faster once kernels are cached. |
| `Cannot load local files without --allowed-local-media-path` during vision summary | vLLM ≥ 0.6 blocks `file://` URLs by default as a security guard | Ensure `allowed_local_media_path="/"` is in the `LLM()` constructor in `build_qwen_engine()`. If you cloned fresh, pull the latest `product_demo_video_analyzer_dgx.py` — it already includes the fix. |

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

Install Python dependencies (macOS / Apple Silicon):

```bash
pip install -r requirements-mac.txt
```

This installs the base pipeline (yt-dlp, OpenCV, Whisper, transcript API, RapidOCR) plus the `openai` client used to reach Ollama. vLLM is intentionally excluded — it is CUDA-only and does not build on macOS; see `requirements-dgx.txt` for the GB10 install.

## LLM Models

| Model | Provider | Role | How to enable |
|---|---|---|---|
| `qwen2.5vl:32b` | Ollama (local) | Text analysis — structured PM/UX/dev report and macro-chunk final summary | Default; override with `--model` |
| `qwen2.5vl:32b` | Ollama (local) | Vision analysis — keyframe+transcript summary, per-frame OCR, macro-chunk window analysis | Default; override with `--vision-model` |
| `gpt-4o` | OpenAI (cloud) | Alternative vision model for per-frame OCR | `--ocr-backend openai`; requires `OPENAI_API_KEY` in `.env` |
| Whisper (`tiny` / `base` / `small` / `medium` / `large`) | OpenAI (runs locally) | Speech-to-text audio transcription fallback | Automatic fallback; size set with `--whisper-model` (default: `base`) |

**Other Ollama vision models** — any Ollama-hosted vision model can replace `qwen2.5vl:32b` via `--vision-model`. Common alternatives:

| Model | Notes |
|---|---|
| `llava` / `llava:13b` / `llava:34b` | LLaVA family; lighter than llama3.2-vision at the 7B size |
| `minicpm-v` | Compact multimodal model, good for OCR-heavy tasks |
| `moondream` | Very small, fast; lower accuracy on complex scenes |
| `bakllava` | LLaVA variant fine-tuned on Mistral |

Pull any model before use: `ollama pull <model-name>`

**Local vs. cloud:** Ollama and Whisper run entirely on your machine — no API key or internet access needed. `gpt-4o` is the only cloud model and requires an `OPENAI_API_KEY`.

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
ollama pull qwen2.5vl:32b
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
# OCR with local RapidOCR (recommended — CPU ONNX, no API key, no GPU load)
python3 product_demo_video_analyzer.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --ocr-backend rapidocr

# OCR with local Ollama vision model
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

When `qwen2.5vl:32b` is memory-constrained or slow, vision requests may time out. The pipeline handles this automatically in two layers:

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

## CLI Reference — `product_demo_video_analyzer_dgx.py` (GB10 / Blackwell)

Authoritative for the DGX pipeline. All options match `parse_args()` in `product_demo_video_analyzer_dgx.py`.

| Option | Default | Description |
|---|---|---|
| `-h`, `--help` | None | Show help text and exit. |
| `--url URL` | — | Video URL (YouTube, Wistia, or any yt-dlp-supported source). Repeat for multiple. At least one of `--url` or `--file` required. |
| `--file PATH` | — | Local video file path. Use alone (Whisper only) or pair 1:1 with `--url` (skips download, URL still used for transcript lookup). |
| `--model MODEL` | `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` | Qwen model ID served by vLLM for transcript-based analysis. |
| `--vision-model VISION_MODEL` | `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` | Qwen model ID for keyframe+transcript vision summary. |
| `--quantization {awq_marlin,nvfp4,fp8,gptq_marlin,none}` | auto-detect | vLLM quantization override. Unset lets vLLM detect from the checkpoint (AWQ → `awq_marlin` on Blackwell). Set to `nvfp4` when you have an NVFP4-quantized checkpoint. |
| `--work-dir DIR` | `artifacts` | Base output directory. |
| `--download-video` | `False` | Keep downloaded video files in `artifacts/videos/`. |
| `--analysis` / `--no-analysis` | Enabled | Enable/disable transcript-only PM/UX/dev report (`<video_id>_analysis.md`). |
| `--vision-summary` / `--no-vision-summary` | Enabled | Enable/disable sparse keyframe multimodal summary (`<video_id>_vision_summary.md`). |
| `--macro-chunking` / `--no-macro-chunking` | **Disabled** | Windowed macro-chunking (~80 LLM calls on a 20-min video). Off by default — enable only for long-form (45+ min) or purely-visual content. |
| `--timeline` / `--no-timeline` | Enabled | Enable/disable timeline outputs (HTML/MD/JSON with frames + transcript). |
| `--timeline-window-seconds N` | `20` | Seconds per timeline window. |
| `--keyframe-mode {interval,scene,ocr}` | **`ocr`** | Keyframe extraction mode. `interval`: one frame every `--keyframe-seconds`. `scene`: one frame per pixel-diff transition. `ocr`: dHash prefilter + RapidOCR text-change detection (default; best for UI-heavy demos). |
| `--keyframe-seconds N` | `20` | For `interval` mode: extract one keyframe every N seconds. |
| `--scene-threshold N` | `25.0` | For `scene` mode: mean pixel difference threshold. Lower = more sensitive. Range 0–255. |
| `--ocr-similarity-threshold F` | `0.85` | For `ocr` mode: text similarity threshold — save a frame when OCR text similarity vs the last saved frame falls **below** this value. Higher = fewer keyframes. |
| `--phash-threshold N` | `5` | For `ocr` mode: Hamming distance (out of 64 bits) below which the dHash prefilter skips OCR entirely. Higher = more skips (fewer OCR calls, coarser keyframes). |
| `--min-keyframe-gap F` | `1.0` | Minimum seconds between saved frames. |
| `--macro-window-seconds N` | `15` | For macro-chunking: window size in seconds. |
| `--capture-fps N` | `6` | For macro-chunking: frame sampling rate (fps). |
| `--macro-frames-per-window N` | `8` | For macro-chunking: max frames per window sent to the vision model. |
| `--frame-width N` | `960` | Resize extracted frames to this max width (pixels). |
| `--max-vision-frames N` | `24` | Max keyframes included in the vision summary. |
| `--whisper-fallback` / `--no-whisper-fallback` | Enabled | Use Whisper if YouTube transcript API and yt-dlp subtitles both fail. |
| `--whisper-model MODEL` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large`. |

### Removed vs. `_analyzer.py`

- `--ocr-backend` — dropped. On the DGX pipeline, OCR is always RapidOCR when `--keyframe-mode ocr` is used. No cloud/Ollama backend needed.

## CLI Reference — `product_demo_video_analyzer.py` (Ollama / legacy)

For the original laptop pipeline. Uses Ollama over HTTP.

| Option | Default | Description |
|---|---|---|
| `-h`, `--help` | None | Show help text and exit. |
| `--url URL` | — | Video URL (YouTube or Wistia). Repeat for multiple. At least one of `--url` or `--file` required. |
| `--file PATH` | — | Local video file path. Use alone (Whisper only) or pair 1:1 with `--url` (skips download, URL still used for transcript lookup). |
| `--model MODEL` | `qwen2.5vl:32b` | Text model for transcript-based PM/UX/dev analysis and macro-chunk final summary. |
| `--vision-model VISION_MODEL` | `qwen2.5vl:32b` | Vision model for image+transcript analysis. |
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
| `--ocr-backend` | `none` | Backend for per-frame OCR text extraction: `rapidocr`, `ollama`, `openai`, or `none`. `rapidocr` runs local ONNX OCR on CPU (~0.4s/frame on Apple Silicon, no API key or GPU — same engine as the DGX build); `ollama`/`openai` use a vision LLM over HTTP. When enabled, visible on-screen text is extracted from each keyframe and included in the timeline, vision summary, and macro-chunk analysis. |
| `--keyframe-mode` | `interval` | Keyframe extraction mode: `interval` (one frame every `--keyframe-seconds`) or `scene` (one frame per distinct screen state, driven by pixel difference). |
| `--scene-threshold` | `25` | Mean pixel difference threshold for `--keyframe-mode scene`. Lower = more sensitive, more frames. Range 0–255. |
| `--min-scene-gap` | `1.0` | Minimum seconds between saved frames in scene-change mode. |
