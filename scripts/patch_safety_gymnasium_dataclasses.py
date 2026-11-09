#!/usr/bin/env python3
"""Fix safety-gymnasium 1.0 np.ndarray dataclass defaults for Python 3.11+."""
from __future__ import annotations
import re, sys
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("safety_gymnasium")
if spec is None or not spec.submodule_search_locations:
    sys.exit("safety_gymnasium not installed")
root = Path(spec.submodule_search_locations[0])
pat = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*:\s*np\.array\s*=\s*(.+?)\s*$", re.M)
n = 0
for p in root.rglob("*.py"):
    txt = p.read_text()
    if "np.array =" not in txt:
        continue
    def repl(m):
        return f"{m.group(1)}{m.group(2)}: np.ndarray = field(default_factory=lambda e=({m.group(3)}): e.copy() if hasattr(e, 'copy') else e)"
    new = pat.sub(repl, txt)
    if new == txt:
        continue
    if "from dataclasses import" in new:
        new = re.sub(
            r"from dataclasses import ([^\n]+)",
            lambda m: m.group(0) if "field" in m.group(1) else f"from dataclasses import {m.group(1)}, field",
            new,
            count=1,
        )
    else:
        new = "from dataclasses import field\n" + new
    p.write_text(new)
    n += 1
    print("patched", p.relative_to(root))
print(f"done ({n} files)")
