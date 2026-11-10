# Frozen protocol

This file is the short operational version of Section V of the manuscript. It
exists so that a re-run can be checked without reading the paper.

## Success criterion

A removal result succeeds when both hold:

1. ASR at most 5%;
2. clean accuracy within two points of the gold retrain for the same seed.

The gold model is deletion with `r = 1` (the oracle forget set). Its clean
accuracies are 88.59%, 88.65%, and 88.77% for seeds 0, 1, and 2.

`r*` is the smallest `p = 1` recall at which the **three-seed mean** meets both
criteria. The paper also reports the stricter **all-seeds** rule (all three
seeds must pass); under it the recipe's tolerance is 0.75 instead of 0.50.

For RadioML, `gamma*` is the minimum recall at which the SNR-averaged ASR meets
the same bound, reported per SNR and method.

## Forget-set construction

`F(r, p)` is built deterministically from the detector score order and the seed:
it contains a fraction `r` of the poison and enough clean samples to reach
precision `p`. Ten cells:

```text
(1.00, 1.0) (0.95, 1.0) (0.90, 1.0) (0.75, 1.0) (0.50, 1.0)
(0.25, 1.0) (0.90, 0.9) (0.90, 0.7) (0.50, 0.9) (0.50, 0.7)
```

Five removal families: deletion and retrain (`retrain_delF`), fine-tune on the
retain set (`finetune_retain`), NegGrad+ (`negrad_plus`), SCRUB (`scrub`), and
the relabel-plus-inoculation recipe (`recipe`). The recipe always uses the true
trigger; it is an upper bound, not a blind defence.

## CIFAR-10

| Item | Setting |
|---|---|
| Data | CIFAR-10, 50k/10k, 10 classes, target class 0 |
| Attacks | BadNets 4x4 checkerboard rows/cols 27-30; Blend `0.8x + 0.2*pattern`, seed 123 |
| Poison rate | 5% (2,500 samples); 10% in E9 |
| Architectures | ResNet-18, VGG-16-BN (CIFAR variants) |
| Optimiser | SGD, momentum 0.9 (Nesterov), weight decay 5e-4 |
| Schedule | OneCycle, max LR 0.1 (VGG-16-BN 0.05), 12 epochs |
| Batch size | 512 |
| Seeds | 0, 1, 2 |
| Detectors | Activation clustering (AC), spectral signatures (SS) |

## RadioML 2016.10a

| Item | Setting |
|---|---|
| Data | 11 modulations, SNR -20 to 18 dB in 2 dB steps |
| Input | 2x128 I/Q windows, unit-energy normalisation |
| Model | three-block 1-D CNN |
| Trigger | last eight samples set to I=0.9, Q=-0.9 |
| Poison | 5% to BPSK |
| Training | 12 epochs, Adam 1e-3 |
| Seeds | 3, with a fixed 80/20 split |

## Block assignments

| Person | Blocks | Driver |
|---|---|---|
| Thota Venkata Sai | E1 seed 0, E3, E6, E7b (BadNets), E9, RadioML | `code/person_sai.py` |
| Sarang N S | E1 seed 1, E7 VGG-16-BN, E8 | `code/person_sarang.py` |
| Divyansh Tiwari | E1 seed 2, E7 ResNet-18/Blend, E7b (Blend), SISA | `code/person_divyansh.py` |
| Urja Mandali | E2 residual poison, E5 durability and MIA, pilot partial deletion | `code/person_urja.py` |

The per-seed E1 assignment is visible in the `person` and `seed` columns of the
run logs. Each driver resumes from existing CSV rows and never overwrites them.

## Block definitions

- **E1** - the ten-cell grid, three seeds, five methods.
- **E2** - residual poison: keep a fraction `f` of the poison in the training
  set and measure ASR, clean accuracy, MIA AUC, and KL.
- **E3** - Calibrate-then-Remove: 20 proxy thresholds from 97.5% to 100%.
- **E5** - durability: 0/5/10/25/50 gradient steps on 100 fresh poison samples,
  plus loss-threshold MIA and clean-versus-trigger KL.
- **E6** - detector operating points at 1%, 5%, and 10% poison (AC and SS).
- **E7** - transfer to VGG-16-BN/BadNets, VGG-16-BN/Blend, ResNet-18/Blend at
  recall 1.0, 0.75, and 0.5.
- **E7b** - Neural-Cleanse-style trigger synthesis, then inoculation with the
  synthesised trigger; IoU against the true mask is logged.
- **E8** - raw feature-space ascent versus the identical schedule with
  batch-norm statistics frozen to the retain set.
- **E9** - 10% poison, simulated recall-0.5 flags, partial label repair versus
  matched clean-removal controls, three seeds.

## Reproducibility checks

```bash
python code/validate_runs.py results/sai_2026-11-02/runs_sai.csv
python code/analysis/audit_numbers.py
```

`validate_runs.py` is a per-batch schema and coverage check. Warnings about
blocks assigned to other drivers are expected. `audit_numbers.py` reads all
frozen CSVs and prints the values used in the manuscript.

## Cost

About 23 GPU-hours on 2xT4 for the CIFAR blocks and about one hour for the radio
block. The four drivers are resumable; a dropped session continues from the last
logged row.
