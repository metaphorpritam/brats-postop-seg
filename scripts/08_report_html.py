#!/usr/bin/env python
"""Render report.md into a single self-contained report.html.

Images are inlined as base64 data-URIs and mermaid fenced blocks become
<pre class="mermaid"> rendered by the mermaid CDN — so the file opens standalone
in any browser (Windows included). Run: uv run --with markdown python scripts/08_report_html.py
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[1]
src = (REPO / "report.md").read_text(encoding="utf-8")

# 1. stash mermaid blocks (they must NOT be markdown-escaped)
mermaids: list[str] = []
def _stash(m: re.Match) -> str:
    mermaids.append(m.group(1).strip())
    return f"\n\nMERMAIDPLACEHOLDER{len(mermaids) - 1}\n\n"
src = re.sub(r"```mermaid\n(.*?)```", _stash, src, flags=re.DOTALL)

# 2. markdown -> html
body = markdown.markdown(src, extensions=["tables", "fenced_code", "sane_lists"])

# 3. restore mermaid as <pre class="mermaid"> (markdown wrapped the placeholder in <p>)
def _restore(m: re.Match) -> str:
    return f'<pre class="mermaid">{mermaids[int(m.group(1))]}</pre>'
body = re.sub(r"<p>MERMAIDPLACEHOLDER(\d+)</p>", _restore, body)
body = re.sub(r"MERMAIDPLACEHOLDER(\d+)", _restore, body)

# 4. inline local images as base64
def _inline(m: re.Match) -> str:
    srcpath = m.group(1)
    p = (REPO / srcpath)
    if p.exists():
        mime = "image/png" if p.suffix.lower() == ".png" else f"image/{p.suffix.lstrip('.').lower()}"
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:{mime};base64,{b64}"'
    return m.group(0)
body = re.sub(r'src="([^"]+)"', _inline, body)

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BraTS Post-Treatment — Achievement Report</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:'neutral', securityLevel:'loose'}});</script>
<style>
 body{{max-width:920px;margin:2rem auto;padding:0 1.2rem;
   font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   line-height:1.65;color:#1b1b1f}}
 h1{{border-bottom:2px solid #e2e2e6;padding-bottom:.4rem}}
 h2{{border-bottom:1px solid #ececf0;padding-bottom:.3rem;margin-top:2.2rem}}
 img{{max-width:100%;height:auto;display:block;margin:1rem auto;border:1px solid #eee;border-radius:6px}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.95rem}}
 th,td{{border:1px solid #d5d5db;padding:6px 11px;text-align:left}}
 th{{background:#f5f5f8}} tr:nth-child(even) td{{background:#fafafb}}
 code{{background:#f2f2f5;padding:2px 6px;border-radius:4px;font-size:.9em}}
 pre{{background:#f7f7f9;padding:1rem;overflow-x:auto;border-radius:8px}}
 pre.mermaid{{background:transparent;text-align:center;border:0}}
 blockquote{{border-left:4px solid #d8d8de;margin:1rem 0;padding:.2rem 1rem;color:#555;background:#fbfbfc}}
 @media (prefers-color-scheme: dark){{
   body{{background:#16161a;color:#e6e6ea}} h1,h2{{border-color:#33333a}}
   th{{background:#242429}} tr:nth-child(even) td{{background:#1d1d22}} th,td{{border-color:#3a3a42}}
   code,pre{{background:#242429}} img{{border-color:#33333a}} blockquote{{background:#1d1d22;border-color:#3a3a42;color:#b8b8c0}}
 }}
</style></head><body>
{body}
</body></html>"""

out = REPO / "report.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({len(html)/1024:.0f} KB, {len(mermaids)} mermaid diagram(s) inlined)")
