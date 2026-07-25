#!/usr/bin/env bash
# Batch 2 — vision-summary + analysis + timeline (NO macro-chunking) on DGX Spark (GB10).
# --qwen-ocr: re-reads the surfaced keyframes with the vision model for accurate OCR text
#   (RapidOCR is gate-only, its garble never surfaced). Adds ~1 vision call per surfaced
#   keyframe — drop the flag for faster runs that carry no OCR text.
# Resumable: skips any video whose _analysis.md already exists (analysis is written last).
# Run from the repo root on edgexpert. Needs: batch2_urls.txt + product_demo_video_analyzer_dgx.py
#
#   nohup ./run_batch2.sh > batch2.log 2>&1 &
#   tail -f batch2.log
#
# DO NOT run while another vLLM batch is active — the pkill guard below kills ALL
# vLLM engines, and two engines would contend for the GPU.
#
# ---------------------------------------------------------------------------
# batch2_urls.txt format — one video per line, two space-separated fields:
#     <video_id> <url>
# e.g.
#     xxxxxxxxxx https://www.youtube.com/watch?v=xxxxxxxxxx
#     yyyyyyyyyy https://www.youtube.com/watch?v=yyyyyyyyyy
#
#   - field 1 (video_id): the ID the analyzer derives from the URL. Used ONLY
#     here for the resume check (artifacts/reports/<id>_analysis.md) — it must
#     match what the script writes.
#   - field 2 (url): passed verbatim to --url; the analyzer downloads from it.
#   ` read -r id url` splits on the first space, so the id must be whitespace-free
#     Blank lines are skipped; add `#` comments at your own risk — this loop does NOT
#     strip them, so keep the file data-only.
# ---------------------------------------------------------------------------

set -u
WORKDIR=artifacts
MANIFEST=batch2_urls.txt
TOTAL=$(grep -c . "$MANIFEST")
n=0

while read -r id url; do
  [ -z "$id" ] && continue
  n=$((n+1))

  if [ -f "$WORKDIR/reports/${id}_analysis.md" ]; then
    echo "[$n/$TOTAL] [skip] $id — analysis already exists"
    continue
  fi

  # Clear any orphaned engine from a prior/killed iteration before launching.
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null && sleep 3

  echo "[$n/$TOTAL] [run]  $id  $url"
  python3 product_demo_video_analyzer_dgx.py \
    --url "$url" \
    --work-dir "$WORKDIR" \
    --vision-summary \
    --analysis \
    --no-macro-chunking \
    --keyframe-mode ocr \
    --phash-threshold 8 \
    --timeline \
    --max-vision-frames 26 \
    --download-video \
    --qwen-ocr
done < "$MANIFEST"

echo "BATCH2 DONE — $(ls "$WORKDIR"/reports/*_analysis.md 2>/dev/null | wc -l) analyses in $WORKDIR/reports/"
