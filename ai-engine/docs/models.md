# Models and local data

Measured on the owner's MacBook (M3, 8 GB) on 2026-11-12.

Every model downloads by itself the first time a feature needs it, then works offline. Nearly all of them
are stored in the shared Hugging Face cache, **`~/.cache/huggingface/hub/`** (one folder per model,
`models--<org>--<name>`). Sizes are what is on disk.

## Models

| Model | Size | Used for | Licence | Status |
|---|---|---|---|---|
| Whisper small, `mlx-community/whisper-small-mlx` (OpenAI) | 0.48 GB | Speech to text: captions, talk-mode clips, voiceover transcripts, voice enrolment | MIT | Used |
| Whisper large-v3-turbo, `mlx-community/whisper-large-v3-turbo` (OpenAI) | 1.61 GB | Checking that cloned Hindi lines say the right words (small mishears Hindi) | MIT | Used |
| CLIP ViT-B/32, `timm/vit_base_patch32_clip_224.openai` (OpenAI, via open_clip) | 0.61 GB | Understanding footage: what is in each shot, matching shots to music edits, voiceover lines and clip titles | MIT | Used |
| Qwen3-4B Instruct, `mlx-community/Qwen3-4B-Instruct-2507-4bit` (Alibaba) | 2.28 GB | Local language model: clip titles, rating talk sections, shot ideas, offline scripts. The default | Apache-2.0 | Used |
| Chatterbox Multilingual v3, `mlx-community/chatterbox-multilingual-v3` (Resemble AI) | 2.71 GB | Voice cloning: speaking scripts in your voice (Hindi + English) | MIT | Used |
| S3Tokenizer V2, `mlx-community/S3TokenizerV2` | 0.49 GB | Part of Chatterbox: turns the reference voice into speech tokens | Apache-2.0 | Used |
| ECAPA speaker verification, `speechbrain/spkrec-ecapa-voxceleb` (SpeechBrain) | 0.09 GB | Checks a cloned line sounds like you; drops other voices from enrolment recordings | Apache-2.0 | Used |
| Qwen2.5-1.5B Instruct, `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | 0.88 GB | Earlier default language model, before Qwen3-4B | Apache-2.0 | Not used by default (still selectable as "fastest") |
| OmniVoice, `mlx-community/OmniVoice-bf16` (k2-fsa) | 1.64 GB | Voice-cloning benchmark only. It lost to Chatterbox: garbled Hindi, 5.6 GB of memory | CC-BY-NC | Not used; safe to delete |

Total: **10.8 GB**.

### Selectable but not downloaded yet

These are offered in the app's language-model menu. Each downloads only if chosen.

| Model | Size | Licence | Note |
|---|---|---|---|
| Qwen2.5-7B Instruct (4-bit) | 4.3 GB | Apache-2.0 | May swap on 8 GB |
| Qwen2.5-3B Instruct (4-bit) | 1.8 GB | Qwen Research, non-commercial | |

## Other local data

| Location | What | Size |
|---|---|---|
| `~/.cache/autocut/ecapa/` | Working copy of the speaker-verification model | ~90 MB |
| `~/.cache/autocut/` (other subfolders) | Saved analysis, so repeat runs are fast (not models). See the breakdown below | ~14 MB |
| `~/Library/Application Support/EI/venv/` | EI's own Python environment and libraries; holds no model weights | — |
| `~/Library/Application Support/EI/settings.json` | App settings, such as the default language model | — |
| `~/Library/Application Support/EI/voices/<name>/` | Voice profiles: reference phrases, recorded takes, speaker fingerprint, room tone | per voice |
| `~/Library/Application Support/EI/styles/<name>.json` | Talking-style profiles learned from transcripts | per style |

What the saved analysis in `~/.cache/autocut/` holds:

| Subfolder | Contents |
|---|---|
| `transcripts` | Speech-to-text results |
| `footage` | Motion, sharpness and brightness per clip |
| `footage-clip` and `clip` | CLIP embeddings |
| `action` | Long-video loudness and motion |
| `music` | Beats and sections |
| `motion` | Motion data |

The analysis cache knows a video by its path, size and modification time. A moved or renamed video is
analysed again.

## Freeing space

* **Delete a model:** remove its folder under `~/.cache/huggingface/hub/`. Removing OmniVoice and Qwen2.5-1.5B
  frees about 2.5 GB.
* **Deleting a model that is still used** is safe; it just downloads again the next time a feature needs it.
* **Clearing `~/.cache/autocut/`** only costs time: videos are analysed again.
* **Never delete** `~/Library/Application Support/EI/voices/` unless you want to remove your voices.
