#!/usr/bin/env python3
"""Draw EI's app icon (original work, MIT OR Apache-2.0) and write an .icns.

    python make_icon.py OUT.icns

A blue rounded tile with block letters "E" and "I" built from rectangles (no font, no image
asset). Needs Pillow (the AI engine's environment has it) and macOS `iconutil`.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

BLUE = (37, 99, 235, 255)
WHITE = (255, 255, 255, 255)


def draw(size: int) -> Image.Image:
    s = 4  # supersample for smooth edges
    n = size * s
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    m = round(n * 0.09)  # macOS icon grid margin
    d.rounded_rectangle([m, m, n - m, n - m], radius=round(n * 0.2), fill=BLUE)
    u = n / 100  # letter grid
    top, bot, stroke = 28 * u, 72 * u, 9 * u
    ex = 27 * u
    d.rectangle([ex, top, ex + stroke, bot], fill=WHITE)  # E stem
    for y, w in ((top, 26), ((top + bot) / 2 - stroke / 2, 22), (bot - stroke, 26)):
        d.rectangle([ex, y, ex + w * u, y + stroke], fill=WHITE)
    ix = 64 * u
    d.rectangle([ix, top, ix + stroke, bot], fill=WHITE)  # I stem
    d.rectangle([ix - 7 * u, top, ix + stroke + 7 * u, top + stroke], fill=WHITE)
    d.rectangle([ix - 7 * u, bot - stroke, ix + stroke + 7 * u, bot], fill=WHITE)
    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "EI.icns")
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "EI.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            draw(base).save(iconset / f"icon_{base}x{base}.png")
            draw(base * 2).save(iconset / f"icon_{base}x{base}@2x.png")
        if shutil.which("iconutil"):
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)
        else:
            draw(1024).save(out.with_suffix(".png"))
            print(f"iconutil not found; wrote {out.with_suffix('.png')}")
            return 0
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
