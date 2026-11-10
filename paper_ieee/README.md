# Manuscript source

`main.tex` is an IEEEtran journal article (10 pt, two-column). The compiled
`main.pdf` is 25 pages with 20 figures, 16 tables, and 73 references.

## Build

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Run twice so cross-references resolve. No BibTeX run is needed; the bibliography
is a manual `thebibliography` block in `sections/refs.tex`. The document uses
only standard packages: IEEEtran, fontenc, mathptmx, amsmath, amssymb, graphicx,
array, booktabs, multirow, algorithm, algpseudocode, balance, url.

On Overleaf, upload the folder and use pdfLaTeX; the source is self-contained.

## Structure

```text
main.tex
sections/
  sec1_intro.tex ... sec13_conclusion.tex
  contributions.tex    author contributions and artifact map
  refs.tex             thebibliography (73 entries)
  bios.tex             author biographies
tables/                audited camera-ready table fragments
figures/               20 vector PDF figures
```

## Audited assets

The table fragments in `tables/` and the figures in `figures/` are the audited
versions used to build `main.pdf`. `code/analysis/build_paper_assets.py`
regenerates raw tables and figures into `generated/` (ignored by git) so a
verification run cannot overwrite the camera-ready assets. To check the numbers
behind the text, run:

```bash
python code/analysis/audit_numbers.py
```

The audit reads the frozen CSVs and prints the gold accuracies, every E1 cell,
the success counts and `r*`, the E2/E3/E6-E9 blocks, the spike, SISA,
fine-pruning, and the per-SNR radio tables.
