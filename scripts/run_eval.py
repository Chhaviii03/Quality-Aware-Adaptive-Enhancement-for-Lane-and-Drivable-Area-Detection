#!/usr/bin/env python
"""
scripts/run_eval.py
-------------------
Command-line entrypoint to run the evaluation harness.

Usage
-----
    python scripts/run_eval.py --model hough --enhancement none --split val
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.run_eval import main

if __name__ == "__main__":
    main()
