#!/usr/bin/env python3
"""
Compatibility entrypoint for the canonical `sr_pondering_machine.py`.

The actively maintained implementation lives in:
`external/sr-pondering-machine.py/sr_pondering_machine.py`

This wrapper keeps legacy root-level commands working:

    python3 sr_pondering_machine.py ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


CANONICAL_SCRIPT = (
    Path(__file__).resolve().parent
    / "external"
    / "sr-pondering-machine.py"
    / "sr_pondering_machine.py"
)


def main() -> None:
    if not CANONICAL_SCRIPT.exists():
        raise SystemExit(
            "[sr_ponder] ERROR: canonical script not found at "
            f"{CANONICAL_SCRIPT}"
        )
    runpy.run_path(str(CANONICAL_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())
