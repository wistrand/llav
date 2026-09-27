# Order Shouldn't Matter

The article on option-order sensitivity of llav's readout: averaging over orders, the spread, and the flip.
The measurements it reports are described in
[agent_docs/experiments/permutation-uncertainty.md](../agent_docs/experiments/permutation-uncertainty.md).

| File                         | Role                                                                                       |
|------------------------------|--------------------------------------------------------------------------------------------|
| `main.tex`                   | The paper: five-page main text and a supplement. Compiles with pdflatex and bibtex         |
| `references.bib`             | Its bibliography                                                                           |
| `order-shouldnt-matter.pdf`  | The current build of `main.tex`                                                            |
| `order-shouldnt-matter.html` | The same text as one page with the figures inlined; the accessible version                 |
| `data/summary.json`          | Every number the paper reports, collected from the raw run outputs by `extract.py`         |
| `tables/*.tex`               | The tables, written from `data/summary.json` by `tables.py`                                |
| `figures/*.svg,pdf`          | The figures, drawn from `data/summary.json` by `figures.py`                                |
| `extract.py`                 | Builds `data/summary.json` from `local/results/perturb/` (not in the repository)           |
| `render.py`                  | Renders `main.tex` as `order-shouldnt-matter.html`; covers the LaTeX subset the paper uses |
| `build.sh`                   | Tables, figures, HTML, PDF (pdflatex if present, else Chrome), page images for checking    |

`./build.sh` rebuilds everything from `data/summary.json`; only `extract.py` needs the raw outputs.
Standard library Python, `rsvg-convert`, and a TeX installation (or Chrome as a fallback for the PDF).
