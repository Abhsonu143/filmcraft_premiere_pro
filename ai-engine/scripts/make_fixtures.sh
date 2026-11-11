#!/bin/sh
# Generate synthetic test media into fixtures/ (gitignored; never commit media).
# Speech comes from macOS `say` with deliberate pauses so silence cutting has work to do.
set -eu
cd "$(dirname "$0")/.."
OUT=fixtures
mkdir -p "$OUT"
LEN=${1:-60}   # seconds of talking-head clip (approx.)

if command -v say >/dev/null 2>&1; then
  say -o "$OUT/talk.aiff" "Hello and welcome back to the channel. [[slnc 1800]] Today I want to show you how this automatic editor works. [[slnc 2500]] Um. [[slnc 1200]] It finds the pauses in my voice and removes them. [[slnc 2000]] The result is a tight edit with jump cuts. [[slnc 900]] Then it adds captions and music. [[slnc 3000]] Thanks for watching, and see you next time."
  say -o "$OUT/narration.aiff" "We arrived in the mountains early in the morning. The air was cold and the sky was clear. [[slnc 600]] Down in the city, the streets were full of cars and people. [[slnc 500]] By the evening we walked along the beach and watched the waves."
else
  echo "macOS 'say' not found; using a tone instead" >&2
  ffmpeg -hide_banner -loglevel error -y -f lavfi -i "sine=f=220:d=20" "$OUT/talk.aiff"
  cp "$OUT/talk.aiff" "$OUT/narration.aiff"
fi

# Talking head: a moving test pattern + the speech, 1920x1080 30 fps, padded to LEN seconds.
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=1920x1080:rate=30" -i "$OUT/talk.aiff" \
  -filter_complex "[1:a]apad=whole_dur=${LEN}[a]" -map 0:v -map "[a]" -t "$LEN" \
  -c:v h264_videotoolbox -b:v 6M -c:a aac -b:a 128k "$OUT/talk.mp4"

# Silent b-roll clips with distinct looks (stand-ins for mountains / city / beach / forest).
# Each carries its subject as a big word: CLIP reads text in images, so matching is testable.
mk() { ffmpeg -hide_banner -loglevel error -y -f lavfi -i "$2,drawtext=text=$4:fontsize=260:fontcolor=white:borderw=8:x=(w-tw)/2:y=(h-th)/2" -t "$3" -an -c:v h264_videotoolbox -b:v 4M "$OUT/$1.mp4"; }
mk mountains "color=c=0x6E8FB5:size=1920x1080:rate=30,drawbox=x=0:y=540:w=1920:h=540:color=white@0.9:t=fill" 12 MOUNTAINS
mk city "mandelbrot=size=1920x1080:rate=30" 12 CITY
mk beach "color=c=0xE8D7A0:size=1920x1080:rate=30,drawbox=x=0:y=0:w=1920:h=400:color=0x3C8DCB:t=fill" 12 BEACH
mk forest "color=c=0x1E5A2A:size=1920x1080:rate=30,noise=alls=40:allf=t" 12 FOREST
# Portrait phone-style clip with a rotation tag (tests rotated footage).
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "testsrc=size=1280x720:rate=30" -t 8 -an -c:v h264_videotoolbox -b:v 3M "$OUT/landscape.mp4"
ffmpeg -hide_banner -loglevel error -y -display_rotation 90 -i "$OUT/landscape.mp4" -c copy "$OUT/phone_rotated.mp4"
rm -f "$OUT/landscape.mp4"

# Music bed: a soft chord (synthetic, no licence issues).
ffmpeg -hide_banner -loglevel error -y -f lavfi -i "sine=f=261.63:d=40" -f lavfi -i "sine=f=329.63:d=40" -f lavfi -i "sine=f=392:d=40" \
  -filter_complex "amix=inputs=3,volume=0.6,tremolo=f=2:d=0.3" -c:a aac -b:a 128k "$OUT/music.m4a"
echo "fixtures in $OUT/"; ls -la "$OUT"
