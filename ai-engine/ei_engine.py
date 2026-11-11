#!/usr/bin/env python3
"""ei_engine: the one entry point the EI app (AI menu) runs. Every subcommand logs progress lines
to stderr ("[autocut] …") and ends by printing one line to stdout:

    EI_RESULT {"ok": true, "outputs": [...], "otio": "...", "srt": "...", "out_dir": "..."}
    EI_RESULT {"ok": false, "error": "..."}

Subcommands (all paths absolute or relative to the current folder):
    doctor                                   what is installed (python packages, ffmpeg features)
    split     VIDEO --out-dir D [--clip-len 60] [--quality same|exact|high|medium|small]
    music     --clips C... --music M --out O.mp4 [--length 30] [--aspect vertical] [--pace medium]
    reference REF --clips C... --out O.mp4 [--aspect same]
    talk      VIDEO --out O.mp4 [--preset long] [--captions] [--music M]
    voiceover VOICE --clips C... --out O.mp4 [--script S] [--music M] [--captions] [--preset travel]

The engine runs as its own process: if it fails or is cancelled, the editor is unaffected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autocut as ac  # noqa: E402
from autocut import AutocutError, log  # noqa: E402


def result(**kw) -> None:
    print("EI_RESULT " + json.dumps(kw), flush=True)


def doctor() -> dict:
    out: dict = {"python": sys.version.split()[0], "executable": sys.executable, "engine": str(Path(__file__).resolve().parent)}
    pk = {}
    for mod in ("numpy", "PIL", "opentimelineio", "mlx_whisper", "open_clip", "torch"):
        pk[mod] = importlib.util.find_spec(mod) is not None
    out["packages"] = pk
    ff = shutil.which("ffmpeg")
    out["ffmpeg"] = ff
    if ff:
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        flt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        out["videotoolbox"] = "h264_videotoolbox" in enc
        out["subtitles"] = " subtitles " in flt
    # What each feature needs: the core features run with numpy alone; captions need Whisper,
    # footage matching needs CLIP (both optional: features degrade, they don't fail).
    out["ready"] = bool(ff) and pk["numpy"]
    out["features"] = {
        "split": bool(ff),
        "music": bool(ff) and pk["numpy"],
        "reference": bool(ff) and pk["numpy"],
        "reference_visual_match": pk["open_clip"] and pk["torch"],
        "talk": bool(ff),
        "captions": pk["mlx_whisper"],
        "voiceover": bool(ff) and pk["mlx_whisper"],
        "otio": pk["opentimelineio"],
    }
    return out


def otio_for(plan_path: Path) -> dict:
    try:
        import otio_export

        o = otio_export.export_plan_file(plan_path)
        srt = o.with_suffix(".srt")
        return {"otio": str(o), **({"srt": str(srt)} if srt.is_file() else {})}
    except AutocutError as e:  # OTIO is a bonus: the video is already rendered
        log(f"OTIO export skipped: {e}")
        return {}


def run(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("split")
    p.add_argument("input")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--clip-len", type=float, default=60.0)
    p.add_argument("--quality", default="same", choices=["same", "exact", "high", "medium", "small"])
    p = sub.add_parser("music")
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--music", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--length", type=float, default=30.0)
    p.add_argument("--aspect", default="vertical", choices=["horizontal", "vertical", "square"])
    p.add_argument("--pace", default="medium", choices=["fast", "medium", "slow"])
    p.add_argument("--music-start", default="auto")
    p.add_argument("--fps", type=float, default=30.0)
    p = sub.add_parser("reference")
    p.add_argument("reference")
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--aspect", default="same", choices=["same", "horizontal", "vertical", "square"])
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--fps", type=float)
    p.add_argument("--no-clip", action="store_true")
    p = sub.add_parser("talk")
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.add_argument("--preset", default="long", choices=sorted(ac.PRESETS))
    p.add_argument("--captions", action="store_true")
    p.add_argument("--music")
    p = sub.add_parser("voiceover")
    p.add_argument("voice")
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--script")
    p.add_argument("--music")
    p.add_argument("--captions", action="store_true")
    p.add_argument("--preset", default="travel", choices=sorted(ac.PRESETS))
    for sp in sub.choices.values():
        sp.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    if a.cmd == "doctor":
        return doctor()
    if a.cmd == "split":
        import split_video

        files = split_video.split(a.input, a.out_dir, a.clip_len, a.quality)
        return {"outputs": files, "out_dir": str(Path(a.out_dir).resolve())}

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("music", "reference"):
        import montage

        a.plan_only = False
        plan = montage.make_music_plan(a) if a.cmd == "music" else montage.make_reference_plan(a)
        plan_path = out.with_name(out.stem + ".montage.json")
        ac.save_plan(plan, plan_path)
        video = montage.render_plan(plan_path, out=str(out), workers=a.workers)
        return {"outputs": [str(video)], "plan": str(plan_path), "out_dir": str(out.parent), **otio_for(plan_path)}
    if a.cmd == "talk":
        plan_path = out.with_name(out.stem + ".plan.json")
        args = ["auto", a.input, "--preset", a.preset, "-o", str(plan_path), "--out", str(out), "--codec", "h264"]
        if a.captions:
            args.append("--captions")
        if a.music:
            args += ["--music", a.music]
        if ac.main(args) != 0:
            raise AutocutError("talking-head edit failed (see the log above)")
        return {"outputs": [str(out)], "plan": str(plan_path), "out_dir": str(out.parent), **otio_for(plan_path)}
    if a.cmd == "voiceover":
        import voiceover_cut as vc

        plan_path = out.with_name(out.stem + ".vo.plan.json")
        args = ["auto", a.voice, "--clips", *a.clips, "--preset", a.preset, "-o", str(plan_path), "--out", str(out), "--workers", str(a.workers)]
        if a.captions:
            args.append("--captions")
        if a.music:
            args += ["--music", a.music]
        if a.script:
            args += ["--script", a.script]
        if vc.main(args) != 0:
            raise AutocutError("voiceover edit failed (see the log above)")
        return {"outputs": [str(out)], "plan": str(plan_path), "out_dir": str(out.parent), **otio_for(plan_path)}
    raise AutocutError(f"unknown command {a.cmd}")


def main(argv: list[str] | None = None) -> int:
    try:
        res = run(sys.argv[1:] if argv is None else argv)
    except AutocutError as e:
        log(f"error: {e}")
        result(ok=False, error=str(e))
        return 1
    except SystemExit as e:  # argparse errors
        if e.code not in (0, None):
            result(ok=False, error="invalid arguments")
        return int(e.code or 0)
    except Exception as e:  # never leave the app waiting without an answer
        log(f"unexpected error: {type(e).__name__}: {e}")
        result(ok=False, error=f"{type(e).__name__}: {e}")
        return 1
    result(ok=True, **res)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
