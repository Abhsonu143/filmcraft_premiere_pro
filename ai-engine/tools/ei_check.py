#!/usr/bin/env python3
"""ei_check: EI's health check from Terminal (the same report as AI ▸ AI Jobs ▸ Check Health).

    python3 ei_check.py [--watch 30] [--clean] [--json]

(Inside the app: /Applications/EI.app/Contents/Resources/ai-engine/tools/ei_check.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import health  # noqa: E402

if __name__ == "__main__":
    sys.exit(health.main())
