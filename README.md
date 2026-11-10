# When the Detector Is Wrong

**A Controlled Study of Post-Training Backdoor Removal Under Imperfect Recall**

Thota Venkata Sai, Sarang N S, Divyansh Tiwari, and Urja Mandali

Department of Electronics and Communication Engineering, Sardar Vallabhbhai
National Institute of Technology (SVNIT), Surat 395007, India

Code, frozen results, and manuscript source:
<https://github.com/venkatasai0852/fml-backdoor-recall-cliff>

This repository contains the code, frozen result tables, notebook generators,
and IEEEtran manuscript source for a measurement study of what happens when a
backdoor removal method is handed the output of an imperfect detector instead of
the true poison set.

## What the paper shows

Post-training removal is normally evaluated with an oracle forget set. In
deployment the forget set comes from a detector, so it has imperfect recall and
precision. We construct forget sets `F(r, p)` with controlled recall `r` and
precision `p`, sweep ten cells and three seeds, and measure five removal
families against a joint criterion: ASR at most 5% **and** clean accuracy within
two points of the gold retrain.

- **Deletion is a cliff, not a slope.** Retraining on the retained set passes
  both halves of the criterion only when the forget set contains every poison
  sample (`r* = 1.00`). At `r = 0.95` the ASR is low but clean accuracy falls to
  84.58%, four points below gold; at `r = 0.90` the backdoor returns at 100%.
- **Recall dominates precision.** At fixed `r = 0.90`, lowering precision from
  1.0 to 0.7 adds 964 clean samples to the forget set and changes deletion's ASR
  from 100.00% to 99.99%. Missing poison is categorically worse than deleting
  extra clean data.
- **Repair tolerates half the poison, but only with the trigger.** Relabelling
  flagged samples and inoculating against the trigger holds ASR at or below 3.6%
  down to `r = 0.5`, at a clean-accuracy cost of 0.7-1.3 points. When the
  trigger is synthesised from the model instead, its IoU with the true patch is
  zero and ASR stays at 98.99% (BadNets) and 95.72% (Blend). The recipe is a
  known-trigger upper bound, not a blind defence.
- **Two refinements fail.** Calibrate-then-Remove never finds a proxy-removed
  model whose trigger response falls below 5% (minimum over twenty thresholds:
  97.5%), and the amplify-and-detect hypothesis does not survive at scale (AUC
  0.468-0.503 against a pre-registered gate of base + 0.10).
- **Apparent unlearning can be a batch-norm artefact.** Feature-space ascent
  reaches ASR 0.18% only by destroying the network; freezing batch-norm
  statistics to the retain set leaves ASR at 99.94-99.98%.
- **Removal suppresses; it does not erase.** The oracle-deleted model returns to
  at least 97.6% ASR within ten gradient steps on 100 fresh poison samples,
  while membership-inference AUC stays at chance and the clean-versus-trigger KL
  gap has vanished.
- **The cliff is not specific to images.** RadioML 2016.10a automatic modulation
  classification shows the same step: deletion needs full recall, while
  trigger-aware repair holds at `r = 0.5` for SNR at or above 0 dB. The minimum
  recall that works is reported as `gamma*` per SNR.

The manuscript is 25 pages in IEEE journal format, with 20 figures, 16 tables,
and 73 references.

## Repository layout

```text
.
|-- code/
|   |-- core_kaggle.py            frozen experimental core (models, attacks,
|   |                             detectors, training loop, CSV logger)
|   |-- person_sai.py             driver: seed-0 grid, E3, E6, E7b, E9, radio
|   |-- person_sarang.py          driver: seed-1 grid, E7 VGG, E8
|   |-- person_divyansh.py        driver: seed-2 grid, E7 Blend, E7b, SISA
|   |-- person_urja.py            driver: E2 residual, E5 probes, pilot blocks
|   |-- make_*.py                 Kaggle notebook generators (A-G, author, RF)
|   |-- nb_common.py              shared notebook helpers
|   |-- smoke_*.py                offline checks used before GPU runs
|   |-- merge_runs.py             merge per-person CSVs into one table
|   |-- validate_runs.py          schema and coverage validator
|   `-- analysis/
|       |-- build_paper_assets.py figure and table builder
|       |-- build_extra_figures.py additional figures
|       |-- audit_numbers.py      independent recomputation of headline numbers
|       `-- fix_legends.py        legend/label post-pass for the figure build
|-- results/                      frozen CIFAR-10 and RadioML run CSVs
|-- urja/                         E2 and E5 CSV tables
|-- spike/                        amplification-pilot CSVs
|-- detection_sai.csv             detector operating points (E6)
|-- fine_pruning_sai.csv          fine-pruning baseline
|-- master_combined_results.csv   pilot-study master table
|-- paper_ieee/
|   |-- main.tex                  IEEEtran journal source
|   |-- sections/                 one .tex per section, refs, bios, contributions
|   |-- tables/                   audited camera-ready table fragments
|   |-- figures/                  20 vector PDF figures
|   `-- main.pdf                  compiled manuscript (25 pages)
`-- docs/
    `-- PROTOCOL.md               frozen protocol, criteria, block assignments
```

The tree keeps the artifact set the manuscript's artifact map cites. Raw
datasets, model checkpoints, generated `.ipynb` files, and intermediate scratch
are excluded; everything needed to recompute or rebuild the paper is included.

## Data

- **CIFAR-10** is downloaded by `torchvision` on first use. On Kaggle, attach a
  CIFAR-10 dataset; `core_kaggle.py` searches `/kaggle/input` for
  `cifar-10-python.tar.gz` or the extracted batches.
- **RadioML 2016.10a** is not redistributed here. On Kaggle, add the public
  dataset `nolasthitnotomorrow/radioml2016-deepsigcom`; the notebook searches
  `/kaggle/input` for `RML2016.10a_dict.pkl`, then `./RML2016.10a_dict.pkl`, then
  `./data/RML2016.10a_dict.pkl`.
- No model checkpoints are distributed. They are large and reproducible from the
  drivers; all result CSVs needed to rebuild every table and figure are included.

## Environment

Tested with Python 3.10 and PyTorch 2.5.1 + CUDA 12.1. The full CIFAR study was
run on Kaggle 2xT4 (about 23 GPU-hours); the radio study takes about one hour.
The code also runs on a 6 GB laptop GPU with the same settings, only slower.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the numbers from the frozen CSVs

No GPU is needed for this path.

```bash
# 1. Validate one driver batch (PASS; coverage warnings for other blocks are expected)
python code/validate_runs.py results/sai_2026-11-02/runs_sai.csv

# 2. Independently recompute every headline number in the paper
python code/analysis/audit_numbers.py

# 3. Regenerate raw tables and figures into paper_ieee/generated/ (scratch)
python code/analysis/build_paper_assets.py
```

`audit_numbers.py` prints the gold accuracies, every E1 cell with per-seed
values, the success counts and `r*`, the E2/E3/E6-E9 blocks, the spike, SISA,
fine-pruning, and the per-SNR radio tables. The committed camera-ready assets in
`paper_ieee/tables/` and `paper_ieee/figures/` are the audited versions used by
the PDF; the builder writes raw regenerations to `paper_ieee/generated/` so it
cannot overwrite them.

## Re-run the experiments

Each author driver is a standalone script that appends one CSV row per run and
resumes from existing rows.

```bash
# on a GPU machine with CIFAR-10 available
python code/person_sai.py        # seed 0: E1, E9, E3, E6, E7b
python code/person_sarang.py     # seed 1: E1, E7, E8
python code/person_divyansh.py   # seed 2: E1, E7, E7b, SISA
python code/person_urja.py       # E2, E5, pilot baselines
```

The notebook generators in `code/make_*.py` wrap the same core and drivers for
Kaggle and are not edited by hand; running them writes the `.ipynb` files. The
radio study is generated by `code/make_RF_study_notebook.py`.

To merge the four CIFAR run logs, copy them into one directory and run:

```bash
mkdir -p results/all_runs
cp results/sai_2026-11-02/runs_sai.csv results/all_runs/
cp results/sarang_2026-11-04/handoff_sarang/runs_sarang.csv results/all_runs/
cp results/sarang_2026-11-05/runs_sarang_v2.csv results/all_runs/
cp results/divyansh_2026-11-03/runs_divyansh.csv results/all_runs/
python code/merge_runs.py results/all_runs results/merged_runs.csv
```

Seed 1 has two logs. The paper's merge rule keeps the original
`runs_sarang.csv` rows except `negrad_plus` and `scrub`, which
`runs_sarang_v2.csv` supersedes after the corrected retain-loop, OneCycle, and
batch-norm handling. `build_paper_assets.py` implements this rule; `results/README.md`
documents it row by row.

## Build the paper

```bash
cd paper_ieee
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

No BibTeX run is needed: the bibliography is a manual `thebibliography` block in
`sections/refs.tex`. The compile is clean (no errors, no undefined references,
no overfull boxes) and produces 25 pages. `paper_ieee/main.pdf` is the compiled
copy. An Overleaf-ready archive is available from the authors.

## Provenance

- `code/core_kaggle.py` is frozen. The four drivers import it and add only
  experiment blocks; they never change seeds, epochs, learning rates, batch
  size, or the split.
- Every result row records the requested and realised recall/precision, forget
  set size, clean accuracy, ASR, wall-clock time, and notes.
- CSVs are never hand-edited. Corrections are new runs, and superseded rows are
  handled by the documented merge rule.
- Every table and figure is written from the CSVs. `audit_numbers.py` recomputes
  the headline claims independently and prints the values quoted in the text.
- Negative and degenerate runs (detector empty sets, collapsed models) are kept
  in the logs rather than filtered out.

## Citation

If you use this code, data, or manuscript, please cite the paper. A
`CITATION.cff` is included; after the repository is published, replace the
placeholder title metadata if the manuscript is accepted at a venue.

## License

Code and result tables are released under the MIT License (see `LICENSE`). The
manuscript text and figures are (c) 2026 the authors. RadioML is governed by its
own license and is not included.

## Contact

Thota Venkata Sai, Sarang N S, Divyansh Tiwari, Urja Mandali - Department of
Electronics and Communication Engineering, SVNIT Surat.
