# Benchmarks

**Machine:** MacBook, Apple M3 (base), 8 GB RAM, macOS 26.6.2. ffmpeg 7.1.1 (Homebrew), Python 3.12.4.
**Date:** 2026-11-12. **Models:** Whisper small (`mlx-community/whisper-small-mlx`), CLIP ViT-B/32 (OpenAI weights) on MPS.

**Media:** synthetic, from `scripts/make_bench_media.sh`: a 1080p30 test pattern with macOS `say`
speech. Real camera/phone footage decodes at a similar hardware speed, but it compresses worse and its
speech is denser, so treat these as a lower bound and re-measure on real projects.

How it was measured: `scripts/bench.py` runs the command, prints each stage as the tool logs it, and
samples memory 5 times a second.
- *tree RSS*: the resident memory of the tool and all its child processes (ffmpeg, model worker).
- *system increase*: how much the whole Mac's used memory rose. This also counts GPU buffers but is noisy
  (other apps, file cache); treat it as an upper bound.

## Talking head: 10:00 clip → 6:25 after cuts (`autocut.py`, preset `long`, captions + music)

| Stage | Cold (first run) | Warm (cached) |
|---|---|---|
| Silence detection (whole file) | 0.6 s | 0.6 s |
| Whisper small, word timestamps | 17.7 s | 0 s (cache hit) |
| **Plan total** | **21.1 s** | **1.0 s** |
| Render: 1 pass, VideoToolbox HEVC 1080p, burned captions, ducked music, loudnorm | 83.6 s | — |
| **End to end** | **≈ 1 min 45 s** | — |

Peak memory: plan 536 MB tree RSS (Whisper in its own process); render 1.08 GB tree RSS (ffmpeg).

## Narration: 10:00 voiceover, 36 min of footage in 12 clips (`voiceover_cut.py`, preset `travel`, captions + music)

| Stage | Cold | Warm |
|---|---|---|
| Whisper small, word timestamps (continuous speech, 1,755 words) | 113.6 s | 0 s |
| CLIP index: ~1,080 frames (1 per 2 s) + 203 beat texts | 257.3 s | 5.8 s (texts only) |
| Beats → 213 shots, matching | < 0.1 s | < 0.1 s |
| **Plan total** | **373 s (6 min 13 s)** | **7 s** |
| Render 213 shots, h264_videotoolbox, 2 workers | 93.2 s | 0 s if the plan didn't change |
| Final mux: concat + burn captions (re-encode) + ducked music | 87.2 s | 87 s |
| **Render total** | **182 s** | — |
| **End to end** | **≈ 9 min 15 s** | — |

Output: exactly 18,000 frames = 600.000 s, the same length as the voice.

Peak memory: plan 970 MB tree RSS (Whisper, then CLIP, one at a time); render 496 MB with 2 workers
and 719 MB with 3.

Workers: 3 parallel encodes took 175.6 s against 182.2 s with 2 (shots 86.9 s against 93.2 s), only 4 % faster
for about 45 % more memory. **Default stays 2.**

## Against the target

A 10-minute video should take 30 minutes or less. Measured: about 2 min (talking head) and about 9 min (narration,
first run). Both are well inside the target, and memory stayed under about 1.1 GB for our processes on an 8 GB Mac.

## Where the time goes (next optimisations)

1. **CLIP indexing (cold)**: ffmpeg decodes *every* frame to sample one every 2 s. Decoding keyframes only
   (`-skip_frame nokey`) or seeking would cut this several-fold. It's a one-time cost per clip (cached).
2. **Caption burn in voiceover mode** re-encodes the whole video in the final step (87 s). Burning
   each shot's slice of captions during the shot encode would make the final step a stream copy.
3. Whisper on continuous speech (113 s for 10 min) is fine; `whisper-base` would be about 2× faster but less
   accurate, and `large-v3-turbo` more accurate but about 3× slower and around 1.6 GB.
