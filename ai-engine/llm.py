"""llm: the local language models EI can use (MLX, on the Apple GPU), and which one is the default.

The default is saved in EI's settings (`~/Library/Application Support/EI/settings.json`,
key `llm_model`; AI ▸ AI Engine Setup changes it). A command can override it with `--llm KEY`.
Models download from Hugging Face the first time they are used, then work offline.
Every call runs in its own process, so the model's memory is freed as soon as it is done.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import autocut as ac
from autocut import log

# key: (Hugging Face repo, size on disk, licence, note)
MODELS: dict[str, tuple[str, str, str, str]] = {
    "qwen3-4b": ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "2.3 GB", "Apache-2.0", "best for 8 GB Macs (default)"),
    "qwen2.5-7b": ("mlx-community/Qwen2.5-7B-Instruct-4bit", "4.3 GB", "Apache-2.0", "largest; may swap on 8 GB, fine on 16 GB"),
    "qwen2.5-3b": ("mlx-community/Qwen2.5-3B-Instruct-4bit", "1.8 GB", "Qwen Research (non-commercial)", "smaller"),
    "qwen2.5-1.5b": ("mlx-community/Qwen2.5-1.5B-Instruct-4bit", "0.9 GB", "Apache-2.0", "fastest"),
}
DEFAULT = "qwen3-4b"


def settings_path() -> Path:
    if os.environ.get("EI_SETTINGS"):
        return Path(os.environ["EI_SETTINGS"])
    return Path.home() / "Library" / "Application Support" / "EI" / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text())
    except (OSError, ValueError):
        return {}


def save_default(key: str) -> None:
    if key not in MODELS and key != "none":
        raise ac.AutocutError(f"unknown model {key!r} ({', '.join(MODELS)}, none)")
    s = load_settings()
    s["llm_model"] = key
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=2))


def resolve(key: str | None) -> str | None:
    """The model key to use: explicit choice, else the saved default, else DEFAULT. None = no model."""
    k = key if key not in (None, "", "default") else load_settings().get("llm_model", DEFAULT)
    if k == "none":
        return None
    return k if k in MODELS else DEFAULT


def _worker(prompts: list[str], repo: str, max_tokens: int) -> list[str]:
    from mlx_lm import generate, load

    model, tok = load(repo)
    outs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        try:  # Qwen3 thinking models: answer directly
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = generate(model, tok, prompt=text, max_tokens=max_tokens, verbose=False)
        outs.append(re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip())
    return outs


def ask(prompts: list[str], key: str | None = None, max_tokens: int = 60) -> list[str] | None:
    """Answers from the chosen model, or None when no model is wanted or it can't run."""
    k = resolve(key)
    if k is None or not prompts:
        return None
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        log("mlx-lm is not installed: no language model (rules only)")
        return None
    repo = MODELS[k][0]
    t0 = time.time()
    try:
        outs = ac.run_isolated(_worker, prompts, repo, max_tokens)
    except Exception as e:  # offline before the first download, out of memory …
        log(f"language model {k} unavailable ({type(e).__name__}: {str(e)[:120]}); using rules")
        return None
    log(f"{k}: {len(prompts)} answers in {time.time() - t0:.1f}s")
    return outs


def clean_title(o: str) -> str | None:
    line = next((x for x in o.strip().splitlines() if x.strip()), "")
    line = re.sub(r"^\s*(title\s*:\s*)", "", line, flags=re.I)
    line = re.sub(r'["“”*#]', "", line).strip().rstrip(".")
    words = line.split()
    return " ".join(words[:8]) if words else None


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="EI's local language models")
    ap.add_argument("--set-default", choices=[*MODELS, "none"])
    a = ap.parse_args(argv)
    if a.set_default:
        save_default(a.set_default)
    cur = load_settings().get("llm_model", DEFAULT)
    print(json.dumps({"default": cur, "models": {k: {"repo": v[0], "size": v[1], "license": v[2], "note": v[3]} for k, v in MODELS.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
