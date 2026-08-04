# HOI x BrainCo pretraining, aligned fingertip frame — final

Generated 2026-08-03. 160 orchestrated runs, 0 failures.

## Setup

| | |
| --- | --- |
| Fingertip frame | HOI converted to BrainCo's wrist frame (`align_xyz_to_brainco_frame`); BrainCo keeps raw FK in pretraining, downstream and inference |
| Position z-score | OFF — see "Why normalization was dropped" |
| HOI pretraining | 200 epochs, lr 4e-4, batch 2048 → 49,200 steps |
| BrainCo pretraining | batch 512, lr 1e-4 (linear scale of HOI's 2048/4e-4) |
| grasp | `task/xyznorm/grasp_temporal_w15_cls_d4_fe4`, 4-fold CV |
| slip | `task/xyznorm/slip_temporal_w15_cls_d4_fe4_v3`, 3-fold CV |
| Seeds | pretraining 42/43/44 x downstream 0/1; `±` is sd over all runs of the arm |
| Metric | balanced accuracy |

## Ablation — Epoch Average

| # | HOI | BrainCo | Model | grasp | slip | n |
| ---: | :---: | :---: | --- | ---: | ---: | ---: |
| 1 | ✗ | ✗ | no pretraining | 0.8396 ± 0.0010 | 0.7072 ± 0.0052 | 2 |
| 2 | ✓ | ✗ | HOI tip | 0.9234 ± 0.0061 | 0.7279 ± 0.0059 | 6 |
| 4 | ✓ | ✗ | HOI jointonly (no force) | 0.8664 ± 0.0032 | 0.5142 ± 0.0048 | 6 |
| 5 | ✗ | ✓ | brainco-only SSL, 31k steps | 0.8624 ± 0.0070 | 0.6920 ± 0.0099 | 6 |
| 5b | ✗ | ✓ | brainco-only SSL, 49k steps | 0.8734 ± 0.0135 | 0.7032 ± 0.0101 | 4 |
| 6 | ✗ | ✓ | brainco-only jointonly | 0.8101 ± 0.0219 | 0.5085 ± 0.0062 | 6 |
| 7 | ✓ | ✓ | **hoi-init SSL** | **0.9334 ± 0.0051** | 0.7211 ± 0.0097 | 6 |
| 8 | ✓ | ✓ | hoi-init gentle | 0.9292 ± 0.0058 | **0.7284 ± 0.0081** | 6 |

## Ablation — Last Epoch

| # | Model | grasp | slip | n |
| ---: | --- | ---: | ---: | ---: |
| 1 | no pretraining | 0.8460 ± 0.0060 | 0.7097 ± 0.0023 | 2 |
| 2 | HOI tip | 0.9305 ± 0.0042 | **0.7316 ± 0.0051** | 6 |
| 4 | HOI jointonly | 0.8802 ± 0.0046 | 0.5199 ± 0.0110 | 6 |
| 5 | brainco-only SSL 31k | 0.8609 ± 0.0096 | 0.6933 ± 0.0155 | 6 |
| 5b | brainco-only SSL 49k | 0.8822 ± 0.0139 | 0.6982 ± 0.0083 | 4 |
| 6 | brainco-only jointonly | 0.8243 ± 0.0160 | 0.5148 ± 0.0143 | 6 |
| 7 | hoi-init SSL | 0.9355 ± 0.0050 | 0.7226 ± 0.0100 | 6 |
| 8 | **hoi-init gentle** | **0.9364 ± 0.0066** | 0.7277 ± 0.0150 | 6 |

## HOI data scaling — constant 49,200 steps

Fraction of the HOI training corpus, 3 pretraining seeds x 2 downstream seeds each.
The subset is repeated inside an epoch so batch, learning rate and step budget are
identical at every point; only data quantity varies. Subsets nest.

| data | files | grasp Avg | grasp Last | slip Avg | slip Last |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 35 | 0.9196 ± 0.0049 | 0.9308 ± 0.0051 | 0.7177 ± 0.0071 | 0.7298 ± 0.0113 |
| 2% | 70 | 0.9248 ± 0.0044 | 0.9323 ± 0.0035 | 0.7109 ± 0.0074 | 0.7167 ± 0.0173 |
| 5% | 176 | 0.9280 ± 0.0045 | 0.9338 ± 0.0048 | 0.7118 ± 0.0124 | 0.7130 ± 0.0095 |
| 10% | 353 | 0.9245 ± 0.0028 | 0.9321 ± 0.0044 | 0.7214 ± 0.0140 | 0.7241 ± 0.0134 |
| 20% | 707 | 0.9261 ± 0.0047 | 0.9341 ± 0.0057 | 0.7259 ± 0.0102 | 0.7309 ± 0.0107 |
| 50% | 1,769 | 0.9316 ± 0.0035 | 0.9388 ± 0.0034 | 0.7173 ± 0.0097 | 0.7262 ± 0.0117 |
| 100% | 3,539 | 0.9234 ± 0.0061 | 0.9305 ± 0.0042 | 0.7279 ± 0.0059 | 0.7316 ± 0.0051 |

## Findings

| | |
| --- | --- |
| **HOI corpus size barely matters** | 1% of the corpus (35 files, ~5k windows) reaches 0.9196 grasp against 0.9234 at 100% — a 0.004 gap against a 0.005 sd. No monotone trend anywhere on the curve; 50% is the nominal best on grasp and 100% on slip, both within noise. At a fixed step budget the benefit of HOI pretraining is not about data volume. |
| Pretraining still buys a lot | Any HOI arm beats no-pretraining by ~8 points on grasp. The gain comes from having *some* hand-interaction pretraining, not from having a lot of it. |
| BrainCo SSL on top of HOI helps slightly | hoi-init SSL leads grasp (0.9334 vs 0.9234 for HOI tip); gentle leads slip on Epoch Average (0.7284 vs 0.7279, a tie). Both are ≤1 point and near the seed spread. |
| BrainCo SSL alone is not competitive | 0.8624 grasp against HOI tip's 0.9234. Raising it to HOI's exact 49,200-step budget (row 5b) recovers ~1 point — 0.8734 — so the gap is data, not optimization. On slip it stays *below* no-pretraining (0.7032 vs 0.7072). |
| Force is what slip needs | Both jointonly arms collapse to ~0.51 on slip, chance for the 6-class problem, while holding 0.81–0.88 on grasp. grasp is largely hand kinematics; slip is not. |
| BrainCo without force is the worst arm | 0.8101 grasp, below no-pretraining's 0.8396. 44 minutes of single-embodiment teleop with no force signal actively hurts. |

## Why per-axis normalization was dropped

Alignment was first paired with a z-score of the position stream. Epoch-controlled
at pretraining epoch 200, seed 0:

| arm | grasp | slip |
| --- | ---: | ---: |
| HOI tip, original pipeline | 0.9342 | 0.7671 |
| HOI tip, aligned + z-scored | 0.9194 | 0.7412 |
| no pretraining, aligned + z-scored | 0.9214 | 0.7415 |

It cost 1.5 / 2.6 points and erased the benefit of pretraining entirely — the
pretrained and untrained arms became indistinguishable. Per-axis whitening also
flattens the ~1.8x palm-normal scale gap between a human hand and the robot hand,
which is real signal. Alignment alone was kept.

## Alignment verification

Checked per finger, not in aggregate. After converting HOI into BrainCo's frame
both corpora agree on axis meaning:

| | HOI (aligned) | BrainCo |
| --- | --- | --- |
| lateral y, thumb − pinky (left / right) | +0.081 / +0.073 | +0.088 / +0.093 |
| length z, index / middle | 0.140 / 0.127 | 0.121 / 0.123 |
| palm-normal x, range | 0.053–0.068 | 0.028–0.041 |

Lateral decreases monotonically thumb → pinky with the same sign on both hands;
index/middle are longest. The residual x gap is a human hand versus a robot hand,
not a permutation error.

## Caveats

- Row 1 has n=2 (one pretraining seed is meaningless without pretraining); every
  other row has n=4–6.
- Fold-to-fold spread ran several times the seed spread in earlier rounds, so
  treat differences under ~0.01 as unresolved. Most gaps in the ablation are at
  or below that.
- HOI arms load 128/131 encoder tensors — `signal_mean`, `signal_std` and
  `sensor_embed.proj.weight` are dropped (proximity-only checkpoints against a
  4-channel task) and the force projection is relearned downstream.
- One downstream backbone lr (1e-4), ID protocol only, no OOD.
- `HOI 42j` was dropped after the first round; its two-seed numbers were 0.9276
  grasp / 0.7249 slip (Epoch Average), within noise of HOI tip.
- The data-scaling curve holds *optimizer steps* constant, not epochs. A
  fixed-epoch design would confound data quantity with optimization budget.
