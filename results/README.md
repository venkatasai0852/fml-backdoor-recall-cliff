# Results

Every CSV here is a frozen record. None was transcribed or edited by hand.
`paper_ieee/sections/contributions.tex` maps files to the paper's experiments.

## CIFAR-10 run logs

| File | Rows | Contents |
|---|---:|---|
| `results/sai_2026-11-02/runs_sai.csv` | 96 | Seed 0: E1 grid, E3, E6, E7b, E9 |
| `results/sarang_2026-11-04/handoff_sarang/runs_sarang.csv` | 69 | Seed 1: E1 v1, E7, E8 raw ascent |
| `results/sarang_2026-11-05/runs_sarang_v2.csv` | 27 | Seed 1: E1 `negrad_plus`/`scrub` corrected, E8 BN-guarded |
| `results/divyansh_2026-11-03/runs_divyansh.csv` | 61 | Seed 2: E1, E7, E7b, SISA |
| `urja/metrics_table_urja.csv` | 20 | E2 residual-poison probes |
| `urja/relearning_summary_urja.csv` | 12 | E5 relearning ASR at steps 0/5/10/25/50 |
| `detection_sai.csv` | 10 | E6 detector operating points |
| `fine_pruning_sai.csv` | 16 | Fine-pruning baseline |
| `spike/spike_repair_seed{0,1}.csv` | 7/5 | Amplification pilot |
| `results/rf_study_2026-11-09/runs_rf_study.csv` | 51 | RadioML run-level results |
| `results/rf_study_2026-11-09/rf_study_per_snr.csv` | 1,020 | RadioML per-SNR cells |
| `results/rf_study_2026-11-09/gamma_star.csv` | 80 | Minimum effective recall per method and SNR |
| `master_combined_results.csv` | 124 | Earlier pilot master table |

These 14 files are exactly the inputs read by
`code/analysis/build_paper_assets.py`; every table and figure in the paper is
rebuilt from them, and `code/analysis/audit_numbers.py` recomputes the headline
numbers from the same files. Run metadata, pre-registered gate summaries,
per-seed pivots, and per-shard SISA logs stay in the author-side run folders and
are not required for the paper build.

## Seed-1 merge rule

Seed 1 has two logs. The paper keeps `runs_sarang.csv` for every E1 row except
`negrad_plus` and `scrub`; those two methods are replaced by
`runs_sarang_v2.csv`, which was rerun after fixing the retain-loop, OneCycle,
and batch-norm-stat handling. Concatenating both logs without this rule creates
52 duplicate keys. `code/analysis/build_paper_assets.py` implements the rule in
`load()`; `code/merge_runs.py` is a plain concatenator and should not be used on
the two seed-1 files together.

## Notes

- Requested and realised recall/precision are both logged. Realised precision is
  stored with integer rounding (for example 0.8999 or 0.7001); the builder snaps
  it to the two-decimal grid.
- Degenerate runs are retained: the detector's empty 10% flag set, the bimodal
  NegGrad+ seeds, and the collapsed clean-accuracy rows.
- The 10% poison amplification runs use simulated recall-0.5 flags, so they are
  logged under E9 rather than E1.
