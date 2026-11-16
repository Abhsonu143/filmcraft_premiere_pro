#!/bin/sh
# 10-minute benchmark media from the small fixtures (run make_fixtures.sh first). Output: fixtures/bench/
set -eu
cd "$(dirname "$0")/../fixtures"
mkdir -p bench
# 10 min talking head: the 40 s talk loop repeated, over a 1080p30 test pattern.
ffmpeg -hide_banner -loglevel error -y -stream_loop 14 -i talk.aiff -t 600 -ar 48000 bench/talk10.wav
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=1920x1080:rate=30" -i bench/talk10.wav -t 600 \
  -c:v h264_videotoolbox -b:v 8M -c:a aac -b:a 128k bench/talk10.mp4
# 10 min narration and 20 min of footage in ten 2-minute clips.
ffmpeg -hide_banner -loglevel error -y -stream_loop 50 -i narration.aiff -t 600 -ar 48000 bench/narration10.wav
for i in 0 1 2 3 4 5 6 7 8 9; do
  src=$(echo "mountains city beach forest talk" | cut -d' ' -f$(( i % 5 + 1 )))
  ffmpeg -hide_banner -loglevel error -y -stream_loop -1 -i "$src.mp4" -t 120 -an -c:v h264_videotoolbox -b:v 6M "bench/clip$i.mp4"
done
ls -la bench
