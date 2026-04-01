#!/usr/bin/env python3
"""
Convenience entrypoint so you can run sweeps from the repo root:

    python sweep.py ...

The real implementation lives in `eval/scripts/sweep.py`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    sweep_path = Path(__file__).parent / "eval" / "scripts" / "sweep.py"
    runpy.run_path(str(sweep_path), run_name="__main__")


if __name__ == "__main__":
    main()

