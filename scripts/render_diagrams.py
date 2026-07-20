#!/usr/bin/env python3
"""Render every note/diagrams/*.mmd to note/figures/diagrams/*.png via mermaid-cli.

Server-side render so each diagram is a static, reviewable PNG (no client-side CDN).
Uses a shared mermaid config (neutral theme, transparent bg, 2x scale for crispness).

Run: python scripts/render_diagrams.py   (needs node + npx; downloads chromium once)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/pritam/code/brats-postop-seg")
SRC = REPO / "note" / "diagrams"
OUT = REPO / "note" / "figures" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

CFG = SRC / "_mermaid_config.json"
CFG.write_text(json.dumps({
    "theme": "neutral",
    "themeVariables": {"fontFamily": "Segoe UI, Helvetica, Arial, sans-serif", "fontSize": "16px"},
    "flowchart": {"htmlLabels": True, "curve": "basis", "nodeSpacing": 45, "rankSpacing": 55, "padding": 12},
}))

mmds = sorted(SRC.glob("*.mmd"))
if not mmds:
    print("no .mmd files"); sys.exit(0)

only = sys.argv[1:] if len(sys.argv) > 1 else None
failed = []
for m in mmds:
    if only and m.stem not in only:
        continue
    out = OUT / f"{m.stem}.png"
    cmd = ["npx", "--yes", "@mermaid-js/mermaid-cli", "-i", str(m), "-o", str(out),
           "-b", "transparent", "-s", "2", "-c", str(CFG)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.exists() and r.returncode == 0:
        print(f"  OK  {m.name} -> {out.relative_to(REPO)}  ({out.stat().st_size/1024:.0f} KB)")
    else:
        failed.append(m.name)
        print(f"  FAIL {m.name}\n    {r.stderr.strip()[-400:]}")
if failed:
    sys.exit(f"failed: {failed}")
print("all diagrams rendered.")
