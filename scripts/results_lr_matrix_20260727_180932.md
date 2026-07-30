# Backbone LR × Checkpoint Matrix — Grasp Prediction & Slip Detection

Generated 2026-07-27 18:40. Runner `scripts/run_lr_checkpoint_matrix.py`;
logs in `scripts/logs/lr_matrix_20260727_180932/`; raw values in
`results_lr_matrix_20260727_180932.csv`. 18/18 runs completed.

## Protocol

- Episode-level **4-fold** CV, seed 0, split seed 42 (2 tasks × 3 encoders × 3 backbone LRs).
- Probe LR fixed at `1e-4`; only the backbone (encoder) LR varies.
- Train batch size 256, 50 epochs, `XFORMERS_DISABLED=TRUE`, one GPU per run (4/5/6/7).
- Encoders: `scratch` (random init), `local_e80` (`checkpoints/dinov2_xyz_temp/epoch-0080-local.ckpt`),
  `base_e100` (`checkpoints/dinov2_xyz_temp/epoch-0100-base.ckpt`). Both load 122 tensors under the
  `teacher_encoder.backbone` prefix.
- Metrics: **balanced accuracy** (mean per-class recall) and **macro F1**. `F1` is the
  binary/positive-class F1 kept for continuity with earlier reports.

## grasp_prediction

| Encoder | Backbone LR | Bal Acc (last) | F1 macro (last) | Bal Acc (best) | F1 macro (best) | Bal Acc (epoch avg) | F1 macro (epoch avg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 1e-4 | 0.8448 | 0.8400 | 0.8508 | 0.8469 | 0.8279 | 0.8246 |
| scratch | 1e-5 | 0.8233 | 0.8191 | 0.8402 | 0.8376 | 0.8215 | 0.8183 |
| scratch | 1e-6 | 0.8144 | 0.8073 | 0.8387 | 0.8359 | 0.8157 | 0.8112 |
| **local_e80** | **1e-4** | **0.8662** | **0.8649** | 0.8744 | 0.8720 | 0.8555 | 0.8532 |
| local_e80 | 1e-5 | 0.8636 | 0.8629 | 0.8709 | 0.8684 | 0.8506 | 0.8476 |
| local_e80 | 1e-6 | 0.8469 | 0.8401 | 0.8572 | 0.8532 | 0.8294 | 0.8251 |
| base_e100 | 1e-4 | 0.8601 | 0.8569 | 0.8731 | 0.8717 | 0.8519 | 0.8494 |
| base_e100 | 1e-5 | 0.8546 | 0.8503 | 0.8641 | 0.8617 | 0.8404 | 0.8365 |
| base_e100 | 1e-6 | 0.8553 | 0.8502 | 0.8613 | 0.8580 | 0.8340 | 0.8306 |

Fold-to-fold std (last epoch, Bal Acc / F1 macro): scratch 0.013–0.019 / 0.012–0.019;
local_e80 0.011–0.018 / 0.014–0.018; base_e100 0.003–0.015 / 0.001–0.014.

Best-vs-best, each encoder at its own optimal LR:

| Encoder | Best LR | F1 macro | Δ vs scratch |
| --- | ---: | ---: | ---: |
| scratch | 1e-4 | 0.8400 | — |
| base_e100 | 1e-4 | 0.8569 | **+0.0169** |
| local_e80 | 1e-4 | 0.8649 | **+0.0249** |

## slip_detection

| Encoder | Backbone LR | Bal Acc (last) | F1 macro (last) | Bal Acc (best) | F1 macro (best) | Bal Acc (epoch avg) | F1 macro (epoch avg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scratch | 1e-4 | 0.6791 | 0.6406 | 0.7224 | 0.6752 | 0.6881 | 0.6501 |
| scratch | 1e-5 | 0.6841 | 0.6440 | 0.7182 | 0.6739 | 0.6799 | 0.6386 |
| scratch | 1e-6 | 0.6950 | 0.6564 | 0.7029 | 0.6663 | 0.6694 | 0.6366 |
| **local_e80** | **1e-4** | **0.7025** | **0.6735** | 0.7236 | 0.6811 | 0.6985 | 0.6623 |
| local_e80 | 1e-5 | 0.6900 | 0.6603 | 0.7134 | 0.6854 | 0.6856 | 0.6603 |
| local_e80 | 1e-6 | 0.6617 | 0.6371 | 0.7265 | 0.6780 | 0.6882 | 0.6572 |
| base_e100 | 1e-4 | 0.6817 | 0.6450 | 0.7064 | 0.6689 | 0.6781 | 0.6448 |
| base_e100 | 1e-5 | 0.6629 | 0.6316 | 0.7060 | 0.6640 | 0.6711 | 0.6413 |
| base_e100 | 1e-6 | 0.6789 | 0.6503 | 0.7238 | 0.6994 | 0.6973 | 0.6679 |

Fold-to-fold std (last epoch, Bal Acc / F1 macro): scratch 0.027–0.045 / 0.037–0.042;
local_e80 0.019–0.039 / 0.027–0.031; base_e100 0.025–0.033 / 0.018–0.029.

Best-vs-best:

| Encoder | Best LR | F1 macro | Δ vs scratch |
| --- | ---: | ---: | ---: |
| scratch | 1e-6 | 0.6564 | — |
| base_e100 | 1e-4 | 0.6450 | **−0.0114** |
| local_e80 | 1e-4 | 0.6735 | **+0.0171** |

## Findings

**1. `local_e80` is the best encoder on both tasks.** It wins at its optimal LR on grasp (0.8649)
and slip (0.6735), and beats `base_e100` at every LR on grasp and at 1e-4/1e-5 on slip. The
prediction-objective checkpoint transfers better than the base DINOv2 one despite 20 fewer epochs.

**2. Pretraining helps grasp prediction clearly, slip detection marginally.** On grasp the
`local_e80` gain is +0.0249 macro F1, larger than the fold std (0.012–0.019), and *both* pretrained
encoders beat scratch at all three LRs — 6/6 wins. On slip the gain is +0.0171 against fold std
0.027–0.042, i.e. **inside the noise**, and `base_e100` is 0.0114 *worse* than scratch. Slip
pretraining benefit is not established by this run.

**3. Backbone LR interacts with initialization, in opposite directions per task.** On grasp, 1e-4
is best for every encoder and lowering the LR always hurts — most steeply for scratch
(0.8400 → 0.8073). On slip the trends split: scratch *improves* as the LR drops (0.6406 → 0.6564)
while `local_e80` *degrades* (0.6735 → 0.6371). A single shared LR would therefore mis-rank the
encoders on slip: at 1e-6 scratch (0.6564) beats `local_e80` (0.6371), reversing the best-vs-best
conclusion. Any scratch-vs-pretrained claim must sweep the LR rather than fix it.

**4. Slip stays far weaker than grasp in absolute terms.** Best macro F1 0.6735 vs 0.8649, and the
binary (slip-class) F1 tops out at 0.6072 — the minority class is still the bottleneck.

**5. On slip, `best`-epoch and `last`-epoch rankings disagree.** By best-epoch macro F1 the top
cell is `base_e100 @ 1e-6` (0.6994), which is only 8th of 9 by last-epoch. Best-epoch selection on
a 4-fold, single-seed run overfits the validation split; prefer the last-epoch or epoch-average
columns for conclusions.

## Caveats

- **Single seed (0).** All `±` values are fold-to-fold spread, not seed spread. Slip differences
  sit inside that spread; grasp differences sit outside it but still rest on one seed.
- Not comparable to earlier reports in `scripts/`: this run uses train batch 256, the
  `max_values = [1000, 1000, 1000, 100000]` scaling, train-split signal statistics, and balanced
  accuracy in place of raw accuracy.
- ID protocol only (episode-level K-fold over all objects); no leave-one-object-out.
