#!/usr/bin/env python3
"""Thin alias: soft z_min mismatch probe via the harden runner + soft config."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_demo_expts.scripts.run_s1_tabletop_harden import main as harden_main


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Inject soft defaults if the caller did not pass --config / --out.
    has_config = any(a == "--config" or a.startswith("--config=") for a in argv)
    has_out = any(a == "--out" or a.startswith("--out=") for a in argv)
    if not has_config:
        argv = ["--config", str(ROOT / "configs" / "s1_tabletop_soft.yaml"), *argv]
    if not has_out:
        argv = ["--out", str(ROOT / "results" / "s1_tabletop_soft.json"), *argv]
    harden_main(argv)


if __name__ == "__main__":
    main()
