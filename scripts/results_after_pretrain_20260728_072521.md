# Downstream results for the hp1 pretraining variants

Generated 2026-07-28 07:40

## Protocol

- ID protocol: episode-level 4-fold CV, seed 0, split seed 42
- Backbone LR 1e-4; probe LR 1e-4; batch 256
- Final `epoch-*.ckpt` of each pretraining run
- Metrics: balanced accuracy and macro F1 (last downstream epoch); `±` is fold-to-fold spread
- The scratch row is a direct seed-0 run with the same command as the pretraining rows and only
  `task.checkpoint_encoder=null` changed. Logs: `scripts/logs/scratch_seed0_20260728/`.
  It reproduces the seed-0 ID scratch numbers in `results_seed_ood_20260727_185607.csv` to four
  decimals, as expected for a fixed seed.

## grasp_prediction

| Pretraining variant | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Δ F1 macro vs scratch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| *scratch* | — | 0.8448 | 0.0190 | 0.8400 | 0.0187 | 0.8408 | — | OK |
| all_pseudo_force_tiny_rope_temporal3_v2_hp1 | 100 | 0.8437 | 0.0050 | 0.8400 | 0.0073 | 0.8435 | +0.0000 | OK |
| all_pseudo_force_tiny_rope_v2_hp1 | 100 | 0.8998 | 0.0257 | 0.8960 | 0.0307 | 0.9012 | **+0.0560** | OK |
| prediction_local_rope_v2_hp1 | 100 | 0.8569 | 0.0110 | 0.8520 | 0.0098 | 0.8522 | +0.0120 | OK |

Best: **dinov2_all_pseudo_force_tiny_rope_v2_hp1** (macro F1 0.8960, bal acc 0.8998).

Only `rope` clears scratch by more than the fold spread (+0.0560 against a fold std of 0.0307).
`temporal3` ties scratch to four decimals, and `prediction_local`'s +0.0120 is close to its own
fold std (0.0098) — neither is separable from random initialization on a single seed.

## slip_detection

| Pretraining variant | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Δ F1 macro vs scratch | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| *scratch* | — | 0.6791 | 0.0288 | 0.6406 | 0.0420 | 0.5761 | — | OK |
| all_pseudo_force_tiny_rope_temporal3_v2_hp1 | 100 | 0.6818 | 0.0300 | 0.6419 | 0.0238 | 0.5818 | +0.0013 | OK |
| all_pseudo_force_tiny_rope_v2_hp1 | 100 | 0.6948 | 0.0224 | 0.6607 | 0.0436 | 0.5918 | +0.0201 | OK |
| prediction_local_rope_v2_hp1 | 100 | 0.6671 | 0.0303 | 0.6196 | 0.0403 | 0.5727 | −0.0210 | OK |

Best: **dinov2_all_pseudo_force_tiny_rope_v2_hp1** (macro F1 0.6607, bal acc 0.6948).

No variant clears scratch here. The best gain (+0.0201) is under half its own fold std (0.0436), and
`prediction_local` lands 0.0210 *below* scratch. On slip detection none of these three pretrainings
is distinguishable from random initialization.

## Caveats

- Single seed; `±` is fold-to-fold spread within each run.
- One backbone LR (1e-4); ID protocol only (no OOD).
- hp1 changes several hyperparameters at once, so a win here does not attribute to any single one.
