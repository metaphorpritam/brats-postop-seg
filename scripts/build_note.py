#!/usr/bin/env python3
"""Assemble the BraTS explainer note: note/sections/*.md -> note/brats_explainer.html.

A single self-contained HTML page with a FIXED, scrollable left-hand table of contents
(scroll-spy highlights the current section; collapses to a toggle on narrow screens).
Markdown (+tables, admonitions, attr_list) with MathJax-rendered LaTeX (pymdownx.arithmatex,
generic) and diagrams/figures inlined as base64 PNGs. Sections are note/sections/*.md in
sorted order; each starts with `## N. Title {#id}`. The `[TOC]` marker (in the front matter)
is lifted out into the sidebar.

Run: uv run --with markdown --with pymdown-extensions python scripts/build_note.py
"""
from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import markdown

REPO = Path("/home/pritam/code/brats-postop-seg")
NOTE = REPO / "note"

_ap = argparse.ArgumentParser(description="Assemble a note/<src> dir into a self-contained HTML.")
_ap.add_argument("--src", default="sections", help="sections subdir under note/ (default: sections)")
_ap.add_argument("--out", default="brats_explainer.html", help="output HTML filename under note/")
_ap.add_argument("--title", default="BraTS Post-Treatment Glioma Segmentation — An Explainer",
                 help="document <title> and browser-tab name")
_args = _ap.parse_args()
SECTIONS = NOTE / _args.src
OUT = NOTE / _args.out
DOC_TITLE = _args.title

# ---- gather sections in order ---------------------------------------------------
files = sorted(SECTIONS.glob("*.md"))
if not files:
    raise SystemExit(f"no section files in {SECTIONS}")
body_md = "\n\n".join(f.read_text(encoding="utf-8").rstrip() + "\n" for f in files)

# ---- markdown -> html -----------------------------------------------------------
md = markdown.Markdown(
    extensions=[
        "tables", "sane_lists", "attr_list", "md_in_html", "footnotes", "toc",
        "admonition", "pymdownx.arithmatex", "pymdownx.superfences",
        "pymdownx.highlight", "pymdownx.details", "pymdownx.tilde", "pymdownx.caret",
    ],
    extension_configs={
        "pymdownx.arithmatex": {"generic": True},
        "toc": {"permalink": "#", "toc_depth": "2-3"},
    },
)
html_body = md.convert(body_md)

# ---- inline local images (figures/diagrams) as base64 ---------------------------
def _inline(m: re.Match) -> str:
    src = m.group(1)
    if src.startswith("data:") or src.startswith("http"):
        return m.group(0)
    p = (NOTE / src) if not src.startswith("/") else Path(src)
    if not p.exists():
        p = (REPO / src)
    if p.exists():
        mime = "image/png" if p.suffix.lower() == ".png" else f"image/{p.suffix.lstrip('.').lower()}"
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'src="data:{mime};base64,{b64}"'
    print(f"  WARN image not found: {src}")
    return m.group(0)
html_body = re.sub(r'src="([^"]+)"', _inline, html_body)

# ---- lift the [TOC] block out into the sidebar ----------------------------------
mtoc = re.search(r'<div class="toc">.*?</div>', html_body, re.DOTALL)
toc_html = mtoc.group(0) if mtoc else "<p>(no contents)</p>"
if mtoc:
    html_body = html_body.replace(mtoc.group(0), "")

n_imgs = html_body.count("data:image/png;base64")
n_secs = len(files)

# ---- CSS (plain string: literal braces) -----------------------------------------
CSS = r"""
:root{ --fg:#1b1b1f; --bg:#ffffff; --muted:#5a5a66; --line:#e2e2e8; --accent:#0b5cad;
       --card:#f7f8fa; --code:#f2f2f5; --sb:300px; }
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.68;color:var(--fg);background:var(--bg);font-size:17px}
/* ---------- fixed scrollable sidebar ---------- */
#toc-sidebar{position:fixed;top:0;left:0;width:var(--sb);height:100vh;overflow-y:auto;
  padding:1.1rem .8rem 3rem;border-right:1px solid var(--line);background:var(--card);
  font-size:.87rem;-webkit-overflow-scrolling:touch}
#toc-sidebar .toc-header{font-weight:700;text-transform:uppercase;letter-spacing:.06em;
  font-size:.72rem;color:var(--muted);padding:.2rem .5rem .6rem;border-bottom:1px solid var(--line);margin-bottom:.5rem}
#toc-sidebar .toc-header a{color:var(--fg);text-decoration:none}
#toc-sidebar ul{list-style:none;margin:0;padding-left:0}
#toc-sidebar ul ul{padding-left:.7rem;font-size:.95em;border-left:1px solid var(--line);margin-left:.55rem}
#toc-sidebar li{margin:.05rem 0}
#toc-sidebar a{color:var(--fg);text-decoration:none;display:block;padding:.24rem .5rem;border-radius:6px;
  border-left:2px solid transparent;line-height:1.35}
#toc-sidebar>.toc>ul>li>a{font-weight:600}
#toc-sidebar ul ul a{font-weight:400;color:var(--muted)}
#toc-sidebar a:hover{background:rgba(11,92,173,.08)}
#toc-sidebar a.active{color:var(--accent);background:rgba(11,92,173,.10);border-left-color:var(--accent);font-weight:700}
#toc-sidebar a.headerlink{display:none}
/* ---------- content ---------- */
#content{margin-left:var(--sb)}
#content-inner{max-width:860px;margin:0 auto;padding:2rem 1.5rem 6rem}
h1{font-size:2rem;line-height:1.25;border-bottom:3px solid var(--line);padding-bottom:.5rem;margin-top:.3rem}
h2{font-size:1.5rem;margin-top:3rem;border-bottom:1px solid var(--line);padding-bottom:.35rem;scroll-margin-top:1rem}
h3{font-size:1.2rem;margin-top:2rem;scroll-margin-top:1rem}
h4{font-size:1.03rem;margin-top:1.4rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
p,li{overflow-wrap:break-word}
img{max-width:100%;height:auto;display:block;margin:1.3rem auto;border:1px solid var(--line);border-radius:8px;background:#fff}
figure{margin:1.4rem 0} figcaption{font-size:.9rem;color:var(--muted);text-align:center;margin-top:.4rem}
table{border-collapse:collapse;width:100%;margin:1.2rem 0;font-size:.94rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left;white-space:nowrap}
th{background:var(--card)} tr:nth-child(even) td{background:var(--card)}
code{background:var(--code);padding:2px 6px;border-radius:4px;font-size:.88em}
pre{background:var(--code);padding:1rem;overflow-x:auto;border-radius:8px} pre code{background:none;padding:0}
blockquote{border-left:4px solid var(--line);margin:1rem 0;padding:.3rem 1rem;color:var(--muted)}
mjx-container{overflow-x:auto;overflow-y:hidden;max-width:100%}
mjx-container[display="true"]{margin:1.1rem 0}
.admonition{border:1px solid var(--line);border-left-width:5px;border-radius:8px;padding:.7rem 1rem;margin:1.2rem 0;background:var(--card);font-size:.96rem}
.admonition-title{font-weight:700;margin:.1rem 0 .5rem;text-transform:uppercase;letter-spacing:.03em;font-size:.82rem}
.admonition.gotcha{border-left-color:#e0a400;background:#fdf7e3} .admonition.gotcha>.admonition-title{color:#8a6400}
.admonition.example{border-left-color:#0b78c4;background:#eef6fc} .admonition.example>.admonition-title{color:#0b5f97}
.admonition.intuition{border-left-color:#1a9e5b;background:#eaf7ef} .admonition.intuition>.admonition-title{color:#137a45}
.admonition.note{border-left-color:#8a8a96} .admonition.cite{border-left-color:#8a8a96;font-size:.9rem}
.headerlink{opacity:0;margin-left:.4rem;font-weight:400;text-decoration:none} h2:hover .headerlink,h3:hover .headerlink{opacity:.4}
/* ---------- mobile toggle ---------- */
#toc-toggle{display:none;position:fixed;top:.7rem;left:.7rem;z-index:40;background:var(--accent);color:#fff;
  border:0;border-radius:8px;padding:.5rem .8rem;font-size:.95rem;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25)}
#toc-backdrop{display:none;position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:30}
@media (max-width:1024px){
  #toc-sidebar{transform:translateX(-102%);transition:transform .22s ease;z-index:35;box-shadow:2px 0 16px rgba(0,0,0,.2);width:290px}
  body.toc-open #toc-sidebar{transform:translateX(0)}
  body.toc-open #toc-backdrop{display:block}
  #content{margin-left:0}
  #content-inner{padding-top:3.4rem}
  #toc-toggle{display:block}
}
@media (prefers-color-scheme: dark){
  :root{ --fg:#e6e6ea;--bg:#15151a;--muted:#a6a6b2;--line:#33333c;--accent:#5aa9ec;--card:#1e1e25;--code:#22222a; }
  img{background:#f4f4f6}
  #toc-sidebar a:hover{background:rgba(90,169,236,.12)}
  #toc-sidebar a.active{background:rgba(90,169,236,.16)}
  .admonition.gotcha{background:#2a2410} .admonition.example{background:#132430} .admonition.intuition{background:#132a1e}
}
"""

# ---- scroll-spy + mobile toggle JS (plain string) -------------------------------
JS = r"""
(function(){
  var links = Array.prototype.slice.call(document.querySelectorAll('#toc-sidebar a[href^="#"]'));
  var map = {}, targets = [];
  links.forEach(function(a){
    var id = decodeURIComponent(a.getAttribute('href').slice(1));
    var el = document.getElementById(id);
    if(el){ map[id]=a; targets.push(el); }
  });
  function spy(){
    var y = window.scrollY + 130, cur = null;
    for(var i=0;i<targets.length;i++){
      if(targets[i].getBoundingClientRect().top + window.scrollY <= y) cur = targets[i];
    }
    links.forEach(function(a){ a.classList.remove('active'); });
    if(cur && map[cur.id]){
      var a = map[cur.id]; a.classList.add('active');
      a.scrollIntoView({block:'nearest'});
    }
  }
  var t=null;
  window.addEventListener('scroll', function(){ if(t) return; t=setTimeout(function(){t=null;spy();},80); }, {passive:true});
  window.addEventListener('load', spy); spy();
  var btn=document.getElementById('toc-toggle'), bd=document.getElementById('toc-backdrop');
  function close(){ document.body.classList.remove('toc-open'); }
  if(btn) btn.addEventListener('click', function(){ document.body.classList.toggle('toc-open'); });
  if(bd) bd.addEventListener('click', close);
  links.forEach(function(a){ a.addEventListener('click', function(){ if(window.innerWidth<=1024) close(); }); });
})();
"""

# ---- shell ----------------------------------------------------------------------
HTML = (
    "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    f"<title>{DOC_TITLE}</title>\n"
    "<script>\nMathJax = { tex: { inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']], tags: 'none' },"
    " options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] } };\n</script>\n"
    "<script src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js\" async></script>\n"
    "<style>" + CSS + "</style></head>\n<body>\n"
    "<button id=\"toc-toggle\" aria-label=\"Contents\">☰ Contents</button>\n"
    "<div id=\"toc-backdrop\"></div>\n"
    "<nav id=\"toc-sidebar\"><div class=\"toc-header\"><a href=\"#\">Contents — jump to a section</a></div>\n"
    + toc_html + "</nav>\n"
    "<main id=\"content\"><div id=\"content-inner\">\n" + html_body + "\n</div></main>\n"
    "<script>" + JS + "</script>\n</body></html>\n"
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}  ({len(HTML)/1024:.0f} KB, {n_secs} sections, {n_imgs} images, "
      f"TOC lifted to sidebar={'yes' if mtoc else 'NO'})")
