# TV Light Sync

AI ▸ **TV Light Sync…** (engine: `ei_engine.py tv-make`, code: `tv_light.py`).

Plays your clip on the big TV in a recording of your room. The TV backlight (Apex / Ambilight style) matches
your clip's colours, done in software instead of with an HDMI sync box. The monitor and laptop stay as
recorded.

## Recordings (analysed once, cached in `~/.cache/autocut/tvlight/`)

- **Orientation** comes from the file's rotation tag: GoPro vertical recordings are 4K with a 90° tag.
  Horizontal recordings make horizontal videos and vertical ones make vertical videos.
- **TV corners** come from a one-time check per recording: AI ▸ TV Light Sync ▸ Check corners.
  - EI suggests them from Apple Vision rectangles over 16 frames.
  - You drag the four corners onto the TV's picture, inside the black frame.
  - An unchecked recording still works with the suggestion.
- **Tracking:** the TV is followed through the whole recording, 10 times a second, by tracking the room
  (SIFT features, never on screens) as the camera moves.
  - The test recording (106 s, handheld) stayed locked 100% of the time.
- **Light learning:** every 0.1 s EI measures the glow in four zones around the TV (left wall, ceiling, right,
  desk) and the colours at the matching edges of what the TV showed.
  - The recordings were made with real HDMI sync, so they teach EI how this room answers each screen colour:
    `glow = A·edge(t − lag) + b` per zone.
  - Measured on GX011572: right 0.95, desk 0.93, top 0.73, left 0.51 (a lamp there); lag 0.1 s.
- **Untouched areas:** the monitor, laptop and lamps are bright in every frame whatever the backlight does,
  so they are found that way and never recoloured.

## Checking and re-checking corners

AI ▸ TV Light Sync ▸ **Check corners** (or **Re-check**) opens its own window, sized to the screen.

- **Drag the corners.** Drag the four corners onto the TV's picture. The corner you last touched shows
  enlarged in a magnifier, and the arrow keys nudge it one pixel at a time.
- **Moment slider + Show This Moment:** look at any moment of the recording. Once the recording has been
  analysed, the corners start where EI tracked the TV at that moment. If the outline drifted there, fix it and
  save: tracking then starts again from that moment.
- **Preview Alignment:** shows six moments across the recording with the tracked outline, so drift is visible
  at a glance. The first time, it analyses the recording.

## TV bezel (optional)

**TV bezel** draws a thin, near-black frame around your clip: 1.2% of its width on the left, top and right, and
twice that at the bottom, like a real TV, with a faint highlight on the inner edge. Use it when you place the
corners on the TV's **outer** edge (the whole TV), so the bezel comes from EI and fits your clip exactly. Leave
it off when the corners sit on the picture inside the real bezel. Engine option: `--bezel 0.012`.

## Several clips

Add as many clips as you like. They run one after another in one background job (AI ▸ AI Jobs shows
"clip 2/5"). The room analysis is shared, so after the first clip each one takes about its render time.

- A clip that can't be made, or is missing, is reported and the rest carry on.
- Videos are saved as `<clip> - on TV <shape>.mp4` in the output folder (default: "EI on TV" next to the clips).
  An existing file is never overwritten.

## Making a video

1. **Predict the glow.** The glow your clip would cause is predicted every 0.1 s: its edge colours, at the
   camera's screen brightness, go through the learned room response.
2. **Choose the moments.** The recording moments with the closest real glow are chosen as one smooth path
   (Viterbi):
   - mostly continuous playback; a held frame is cheap
   - a jump costs more the more the camera view differs, and less on a cut in your clip
   - every stretch is at least 1 s
3. **Relight.** The remaining difference is closed by recolouring the glow near each TV edge towards the
   predicted colour, weaker with distance. Screens and lamps are left alone.
4. **Place the clip.** Your clip goes on the TV with the tracked perspective. It is matched to how the camera
   saw that screen (its brightness levels), slightly softened, and given camera grain.

The best recording of the chosen shape is picked automatically (lowest light difference).

## Steady picture, framing and a bright screen (November 2026)

- **Steady.** The TV is now found in every frame, not 10 times a second: the room's motion from frame
  to frame (optical flow on the room, never on screens or lamps), held to the accurate 10-a-second
  track. The camera path is smoothed (0.6 s) and every frame is moved onto it, so handheld shake goes
  and the clip stays locked to the TV. This also removes the judder of the 50 fps recordings being
  shown at 30 fps (each output frame stepped one or two recorded frames, unevenly).
- **Framing.** The video shows the part of the recording around the TV: vertical videos let the TV
  fill about 95% of the width with its centre half way down (the lit wall above it, less of the dark
  floor), horizontal ones about 68%. The TV is never cut: when it is at the edge of the recording,
  the framing stops there. The crop also hides the edges the stabilisation moves.
- **Screen.** The clip keeps its own blacks and contrast; only its mid-tones move part of the way to
  how the camera saw this screen, then it is made as bright as a screen in a dark room, with a soft
  bloom onto what is around the TV. Softening is minimal, and grain (fresh every frame) is only on the
  screen; the room keeps the camera's own noise (the old repeating grain made the picture shimmer).

## Measured (M3 8 GB)

| Run | Result |
|---|---|
| 20 s game clip on GX011572 (horizontal) | light difference 1.73, against 3.07 when playing the recording from the start; 5 jumps |
| Analysis | 3.5 min the first time per recording, then cached |
| Render | about 7.5 frames/s, so 80 s for 20 s at 1080p |
| Vertical (GX011576, automatic corners) | works. The automatic corners were slightly off, which the corner check fixes. The light model fits less well on this recording (0.53) |

## Limits / ideas

- **Lens curvature:** the GoPro wide lens bends the TV edges slightly. A single perspective (four corners)
  leaves a pixel or two at the edge middles. Recording in GoPro **Linear** lens mode avoids this; curved-edge
  warping is a possible improvement.
- **Corner guesses:** the automatic corner guess is right on about 5 of 7 recordings, so check the corners
  once.
- **Colours the recordings never showed:** relighting can recolour only so far (gain 0.35–2.8). A colour far
  from anything in the recordings (deep green when the recordings were all warm) matches less well. Recording
  a few minutes of very varied content (colour bars, saturated scenes) gives EI more light to choose from.
