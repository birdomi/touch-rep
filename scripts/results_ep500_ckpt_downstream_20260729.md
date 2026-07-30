# ep500 pretraining snapshots — grasp prediction & slip detection

Generated 2026-07-29 10:45

Two mid-training snapshots of the 500-epoch pretraining runs, evaluated on both
downstream tasks.

| Tag | Pretraining run | Checkpoint |
| --- | --- | --- |
| `ibotfix` | `dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048_ep500` | `2026.07.28-17-40/checkpoints/epoch-0330.ckpt` |
| `recon` | `dinov2_recon_all_pseudo_force_tiny_rope_v2_hp1_tip_s1_b2048_ep500` | `2026.07.28-14-52/checkpoints/epoch-0370.ckpt` |

## Protocol

- ID protocol: episode-level 4-fold CV, seed 0, split seed 42
- Backbone LR 1e-4; probe LR 1e-4; 50 downstream epochs
- Encoder loaded from `teacher_encoder.backbone` (124 tensors) in both cases
- `±` is fold-to-fold spread, not a seed spread

## grasp_prediction

Scratch reference (3 seeds, same settings): **0.8467 ± 0.0058** macro F1.

| Checkpoint | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Best F1m | EpochAvg F1m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recon | 370 | 0.9374 | 0.0095 | **0.9368** | 0.0094 | 0.9383 | 0.9419 | 0.9181 |
| ibotfix | 330 | 0.9309 | 0.0076 | 0.9299 | 0.0069 | 0.9332 | 0.9455 | 0.9120 |

Both are far above scratch (+0.083 / +0.090 macro F1). `recon` edges out
`ibotfix` by 0.007, which is inside the fold spread.

## slip_detection

Scratch reference (3 seeds, same settings): **0.6463 ± 0.0052** macro F1.

| Checkpoint | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Best F1m | EpochAvg F1m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recon | 370 | 0.6703 | 0.0366 | **0.6412** | 0.0521 | 0.5657 | 0.6599 | 0.6317 |
| ibotfix | 330 | 0.6373 | 0.0381 | 0.6059 | 0.0286 | 0.5283 | 0.6198 | 0.6044 |

Neither beats the scratch reference on macro F1. `recon` is level with it
(-0.005, well inside its own 0.052 fold spread); `ibotfix` is below it (-0.040).
Balanced accuracy is higher than scratch for `recon`, so the gap is driven by
the minority (slip) class F1 — `f1bin` 0.57 / 0.53.

## slip_detection, 15-frame windows

Same protocol, `input_window_frames`/`stride` 15 instead of 3
(`.../slip_detection/dinov2_all_rope_w15`). A scratch run under the identical
config is included because w15 labelling differs from the 3-frame protocol, so
the 0.6463 3-frame scratch reference is not a valid baseline here.

| Run | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Best F1m | EpochAvg F1m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **scratch** | – | 0.6904 | 0.0237 | **0.6581** | 0.0286 | 0.5853 | 0.6606 | 0.6302 |
| recon | 370 | 0.6554 | 0.0327 | 0.6332 | 0.0387 | 0.5442 | 0.6458 | 0.6224 |
| ibotfix | 330 | 0.6525 | 0.0371 | 0.6304 | 0.0243 | 0.5390 | 0.6281 | 0.6133 |

Scratch is ahead of both pretrained checkpoints on every column. The best
pretrained run trails it by 0.025 macro F1, which is inside the fold spread
(0.029–0.039), so the gap is not separated by this evidence alone — but the
direction matches the 3-frame result, where scratch also matched or beat both.

`recon` and `ibotfix` are level here (0.6332 vs 0.6304, far inside spread); the
0.035 recon advantage seen at 3 frames does not reproduce at 15.

## Reading across the two tasks

Pretraining helps grasp prediction a lot (+0.083–0.090 macro F1 over scratch)
and does not help slip detection at either window length. Whatever the
fingertip-only DINO objective learns transfers to the grasp task but not to
slip, which stays near 0.63–0.66 macro F1 regardless of initialisation.

## Caveats

- **Both pretraining runs were still training** when these snapshots were taken
  (~17 h in, epoch 330/370 of 500). These are not final checkpoints.
- Single downstream seed; `±` is fold-to-fold spread within one run.
- One backbone LR (1e-4); ID protocol only, no OOD.
- The first slip attempt (GPUs 4/5, logs under
  `scripts/logs/after_pretrain_20260729_101945/`) was killed by an external
  SIGTERM at fold 3/4 and was rerun from scratch on GPUs 0/1 with the
  checkpoints pinned — auto-resolution had drifted to `epoch-0380` for `recon`.

## Artifacts

- grasp: `scripts/results_after_pretrain_20260729_101945.csv`, logs in
  `scripts/logs/after_pretrain_20260729_101945/`
- slip (3-frame): `scripts/results_slip_pinned_20260729_103205.csv`, logs in
  `scripts/logs/slip_pinned_20260729_103205/`
- slip (15-frame): `scripts/results_slip_w15_20260729_105225.csv`, logs in
  `scripts/logs/slip_w15_20260729_105225/`
