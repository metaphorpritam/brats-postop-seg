#!/usr/bin/env bash
# Refresh the GitHub Pages site in docs/ from the current sources. Idempotent.
#
#   docs/index.html     landing page (hand-authored; not regenerated here)
#   docs/explainer.html <- note/brats_explainer.html   (scripts/build_note.py)
#   docs/report.html    <- report.html                 (scripts/08_report_html.py)
#   docs/.nojekyll      stops GitHub Pages running Jekyll over the HTML
#
# Run after changing note/sections/*, report.md, or any figure:
#   bash scripts/publish_docs.sh
set -e
cd "$(dirname "$0")/.."

echo "==> rebuilding the explainer note"
uv run --with markdown --with pymdown-extensions python scripts/build_note.py

echo "==> rebuilding the CV-defense dossier"
uv run --with markdown --with pymdown-extensions python scripts/build_note.py \
  --src dossier --out cv_dossier.html --title "CV Defense Dossier — BraTS Controlled A/B"

echo "==> rebuilding the technical report"
uv run --with markdown python scripts/08_report_html.py

mkdir -p docs
touch docs/.nojekyll
cp note/brats_explainer.html docs/explainer.html
cp note/cv_dossier.html docs/dossier.html
cp report.html docs/report.html

echo "==> docs/ refreshed:"
for f in docs/index.html docs/explainer.html docs/dossier.html docs/report.html; do
  printf '    %-22s %s\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)"
done
echo "    .nojekyll present: $([ -f docs/.nojekyll ] && echo yes || echo NO)"
echo "Commit docs/ and push; GitHub Pages serves from branch 'main', folder '/docs'."
