#!/bin/bash
# Build the paper: figures from the result summaries, HTML from main.tex, and the PDF: through pdflatex and
# bibtex when TeX is installed (latex-build/), else by printing the HTML with Chrome. Page images for
# checking go to pages/.
set -eu
cd "$(dirname "$0")"
python3 tables.py
python3 figures.py
for f in figures/*.svg; do
  rsvg-convert -f pdf -o "${f%.svg}.pdf" "$f"
done
python3 render.py order-shouldnt-matter.html
if command -v pdflatex >/dev/null; then
  mkdir -p latex-build
  tex() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory latex-build main >/dev/null; }
  tex; (cd latex-build && BIBINPUTS=.. bibtex main >/dev/null); tex; tex; tex
  grep -E "Overfull|Undefined|multiply" latex-build/main.log || true
  cp latex-build/main.pdf order-shouldnt-matter.pdf
else
  google-chrome-stable --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/order-shouldnt-matter.pdf" "file://$PWD/order-shouldnt-matter.html" 2>/dev/null
fi
# Local preview of the site: docs/paper/ is gitignored; the Pages workflow copies the outputs when publishing.
mkdir -p ../docs/paper && cp order-shouldnt-matter.pdf order-shouldnt-matter.html ../docs/paper/
rm -rf pages && mkdir pages && pdftoppm -r 70 -png order-shouldnt-matter.pdf pages/page
echo "order-shouldnt-matter.pdf: $(pdfinfo order-shouldnt-matter.pdf | awk '/^Pages/{print $2}') pages"
