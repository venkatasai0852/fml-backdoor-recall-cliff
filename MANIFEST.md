# Manifest

Snapshot of the artifact tree. Sizes are bytes; the short hashes are the first
16 hex characters of SHA-256 and are provided for the files behind the paper's
headline numbers.

## Counts

| Directory | Files | Size |
|---|---:|---:|
| `code/` | 19 | 264 KB |
| `results/` | 8 | 73 KB |
| `paper_ieee/` | 54 | 2.18 MB |
| `docs/` | 1 | 4.4 KB |
| repository total | 97 | 2.55 MB |

The repository total includes the root-level CSVs and requirements files,
`urja/`, `spike/`, `data/README.md`, the license, citation file, and these
docs. Counts exclude LaTeX build products and `paper_ieee/generated/`.

## Key files

| File | Bytes | SHA-256 (16) |
|---|---:|---|
| `paper_ieee/main.pdf` | 1,141,815 | `73ec0a512c60d8e3` |
| `results/sai_2026-11-02/runs_sai.csv` | 10,974 | `4be19744a457b192` |
| `results/sarang_2026-11-04/handoff_sarang/runs_sarang.csv` | 6,510 | `38e5fb281ddd18c2` |
| `results/sarang_2026-11-05/runs_sarang_v2.csv` | 4,371 | `387bd869361e4b3e` |
| `results/divyansh_2026-11-03/runs_divyansh.csv` | 5,931 | `e2e02812f5f0c0c5` |
| `results/rf_study_2026-11-09/runs_rf_study.csv` | 5,643 | `40d478bf5a6d54ae` |
| `results/rf_study_2026-11-09/rf_study_per_snr.csv` | 36,302 | `69ac54cf1abc6262` |
| `results/rf_study_2026-11-09/gamma_star.csv` | 2,253 | `e5c15c0b340fcd33` |
| `urja/metrics_table_urja.csv` | 1,757 | `c2c768c93cd29d95` |
| `urja/relearning_summary_urja.csv` | 1,046 | `30bdbfef79e9824f` |
| `detection_sai.csv` | 706 | `4a0692ce240d9bbc` |
| `fine_pruning_sai.csv` | 653 | `8abcfc2f8a47bedb` |
| `master_combined_results.csv` | 20,768 | `8eda7b4ffc94dd60` |

## Paper assets

- `paper_ieee/main.tex` - IEEEtran journal source.
- `paper_ieee/sections/` - 16 files: 13 sections, `contributions.tex`,
  `refs.tex` (73 entries), `bios.tex`. Two table environments are inline in the
  section files.
- `paper_ieee/tables/` - 14 audited table fragments (16 tables in the PDF).
- `paper_ieee/figures/` - 20 vector PDF figures.
- `paper_ieee/main.pdf` - 25 pages, compiled from the files above.

## Code assets

- Frozen core: `code/core_kaggle.py`.
- Four drivers: `code/person_{sai,sarang,divyansh,urja}.py`.
- Notebook generators: `code/make_{sai,sarang_fix,divyansh_fix,team,RF_study}_notebook.py`,
  plus `code/nb_common.py`.
- Offline checks: `code/smoke_core.py`, `code/smoke_rf_study.py`.
- Merge and validation: `code/merge_runs.py`, `code/validate_runs.py`.
- Analysis: `code/analysis/{audit_numbers.py,build_paper_assets.py,build_extra_figures.py,fix_legends.py}`.

## Result tables

See `results/README.md` for the row-by-row map, including the seed-1 merge rule
(keep the original log except `negrad_plus` and `scrub`, which the v2 log
supersedes). `code/analysis/build_paper_assets.py` reads exactly the 14 CSVs
listed there; no other result file is needed to rebuild the tables and figures.
