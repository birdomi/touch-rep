# Downstream performance vs. pretraining epoch — `dinov2_prediction_local`

Generated 2026-07-27 21:49. Runner `scripts/run_pretrain_epoch_sweep.py`; logs in
`scripts/logs/epoch_sweep_20260727_211810/`; raw values in
`results_epoch_sweep_20260727_211810.csv`. **18/18 runs completed, 0 failed.**

## Protocol

- Pretraining run: `experiments/dinov2_prediction_local_all_pseudo_force_tiny_rope_temporal3/2026.07.26-02-40`
  (still training on GPU 3 during this sweep; `last.ckpt` excluded as a moving target).
- Checkpoints: `epoch-0010` … `epoch-0090` (9 of them).
- ID protocol: episode-level 4-fold CV, seed 0, split seed 42.
- Backbone LR 1e-4, probe LR 1e-4, batch 256, 50 downstream epochs.
- Metrics: balanced accuracy and macro F1 at the last downstream epoch; `±` is fold-to-fold spread.
- **Scratch reference** (same settings, 3 seeds, from `results_seed_ood_20260727_185607.md`):
  grasp macro F1 **0.8467 ± 0.0058**, slip macro F1 **0.6463 ± 0.0052**.

## grasp_prediction

| Pretrain epoch | Bal Acc | ± (fold) | F1 macro | ± (fold) | F1 (bin) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.8553 | 0.0183 | 0.8532 | 0.0183 | 0.8553 |
| 20 | 0.8634 | 0.0207 | 0.8588 | 0.0229 | 0.8597 |
| 30 | 0.8661 | 0.0132 | 0.8634 | 0.0149 | 0.8697 |
| 40 | 0.8583 | 0.0159 | 0.8552 | 0.0182 | 0.8598 |
| 50 | 0.8591 | 0.0172 | 0.8566 | 0.0183 | 0.8597 |
| 60 | 0.8496 | 0.0221 | 0.8478 | 0.0217 | 0.8529 |
| 70 | 0.8685 | 0.0209 | 0.8653 | 0.0197 | 0.8658 |
| 80 | 0.8662 | 0.0141 | 0.8649 | 0.0140 | 0.8648 |
| **90** | **0.8740** | 0.0071 | **0.8720** | 0.0061 | 0.8754 |

Range 0.8478–0.8720 (span 0.0242); mean fold std 0.0171.
Epoch correlation: Pearson r = **+0.555** (p = 0.121), Spearman r = +0.600 (p = 0.088).
First half (10–40) mean 0.8577 vs second half (60–90) mean 0.8625.

## slip_detection

| Pretrain epoch | Bal Acc | ± (fold) | F1 macro | ± (fold) | F1 (bin) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.7092 | 0.0179 | 0.6723 | 0.0355 | 0.6111 |
| 20 | 0.6960 | 0.0312 | 0.6626 | 0.0322 | 0.5986 |
| **30** | 0.7082 | 0.0242 | **0.6773** | 0.0223 | 0.6098 |
| 40 | 0.6855 | 0.0074 | 0.6538 | 0.0226 | 0.5843 |
| 50 | 0.6943 | 0.0424 | 0.6657 | 0.0393 | 0.5901 |
| 60 | 0.6968 | 0.0448 | 0.6681 | 0.0472 | 0.5970 |
| 70 | 0.6946 | 0.0381 | 0.6635 | 0.0280 | 0.5940 |
| 80 | 0.7025 | 0.0190 | 0.6735 | 0.0269 | 0.6072 |
| 90 | 0.6818 | 0.0304 | 0.6500 | 0.0317 | 0.5819 |

Range 0.6500–0.6773 (span 0.0273); mean fold std 0.0317.
Epoch correlation: Pearson r = **−0.355** (p = 0.349), Spearman r = −0.250 (p = 0.516).
First half (10–40) mean 0.6665 vs second half (60–90) mean 0.6638.

## Findings

**1. Every checkpoint beats scratch on both tasks — even epoch 10.** Grasp margins run
+0.0011 (epoch 60) to +0.0253 (epoch 90); slip margins +0.0037 (epoch 90) to +0.0310 (epoch 30).
The benefit of pretraining appears almost immediately and does not need a long schedule.

**2. Grasp improves slowly with pretraining length; the trend is suggestive, not established.**
Pearson r = +0.555 but p = 0.121, and the total span (0.0242) is only ~1.4× the mean fold std
(0.0171). Epoch 90 is the best point and has the tightest folds (±0.0061), so extending the run is
worthwhile — but a monotone claim is not supported by 9 single-seed points.

**3. Slip saturates immediately and shows no epoch dependence.** Epoch 10 (0.6723) is already
indistinguishable from the best point (epoch 30, 0.6773), the correlation is *negative* and far
from significant (r = −0.355, p = 0.349), and the entire 9-checkpoint span (0.0273) is **smaller
than the mean fold std (0.0317)**. The curve is flat noise around ~0.665; more pretraining buys
nothing measurable here.

**4. The two tasks disagree about which checkpoint is best.** Grasp peaks at epoch 90, slip at
epoch 30 — and epoch 90 is slip's *worst* point. There is no single best checkpoint; if one must
serve both, epoch 80 is the most balanced (grasp 0.8649, slip 0.6735 — near-best on both).

**5. This reproduces the earlier `local_e80` numbers exactly.** Epoch 80 here gives grasp 0.8649
and slip 0.6735, matching the seed-0 values from the LR matrix run, which confirms the pipeline is
deterministic under a fixed seed and split.

## Caveats

- **Single seed (0).** `±` is fold-to-fold spread within each run, not seed spread. Prior runs put
  ID seed spread at 0.003–0.006, so seed noise is small relative to the fold noise shown here — but
  the epoch-to-epoch differences on slip sit inside the fold noise regardless.
- One backbone LR (1e-4) only; the LR sweep showed slip rankings can flip at 1e-6.
- The pretraining run was still in progress, so epochs beyond 90 may change the grasp trend.
- ID protocol only; no OOD in this sweep.
