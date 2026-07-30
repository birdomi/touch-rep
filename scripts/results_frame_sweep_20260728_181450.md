# Frame-length sweep — slip detection & grasp prediction

Generated 2026-07-28 20:44. 48/48 runs OK.

## Protocol

- Episode-level 4-fold CV, seeds 0,1, split seed 42
- Probe `BraincoGraspRoPEProbe` (depth 2, 3 heads); backbone LR 1e-4, probe LR 1e-4
- **`balanced_sampling` off, `require_uniform_label` off** (windows may straddle a slip boundary; the label is the last frame's)
- slip: `input_window_frames` = `input_window_stride` = the frame count; grasp: `window_time` = frames / 100 Hz
- Encoders:
  - `tip_s3` — `experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip/2026.07.28-13-23/checkpoints/epoch-0100.ckpt`
  - `tip_s1b2048` — `experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048/2026.07.28-14-48/checkpoints/epoch-0060.ckpt`
  - `recon_tip_s1b2048` — `experiments/dinov2_recon_all_pseudo_force_tiny_rope_v2_hp1_tip_s1_b2048/2026.07.28-14-49/checkpoints/epoch-0060.ckpt`
- Metrics: macro F1 at the **last** downstream epoch and the **EpochAvg** over epochs 10/20/30/40/50. `±` is the seed-to-seed spread; per-run fold std is in the CSV.

## slip_detection

### Last epoch — macro F1

| Encoder | 1 frames | 5 frames | 15 frames | 30 frames |
| --- | ---: | ---: | ---: | ---: |
| scratch | 0.6641 ± 0.0004 | 0.6614 ± 0.0123 | 0.6473 ± 0.0030 | 0.6658 ± 0.0069 |
| tip_s3 | 0.6562 ± 0.0107 | 0.6657 ± 0.0129 | 0.6756 ± 0.0035 | 0.6600 ± 0.0211 |
| tip_s1b2048 | 0.6367 ± 0.0052 | 0.6508 ± 0.0006 | 0.6542 ± 0.0111 | 0.6401 ± 0.0193 |
| recon_tip_s1b2048 | 0.6488 ± 0.0111 | 0.6456 ± 0.0174 | 0.6540 ± 0.0067 | 0.6476 ± 0.0239 |

Δ vs scratch:

| Encoder | 1 frames | 5 frames | 15 frames | 30 frames |
| --- | ---: | ---: | ---: | ---: |
| tip_s3 | -0.0079 | +0.0043 | +0.0282 | -0.0058 |
| tip_s1b2048 | -0.0274 | -0.0106 | +0.0068 | -0.0257 |
| recon_tip_s1b2048 | -0.0152 | -0.0158 | +0.0066 | -0.0182 |

### EpochAvg — macro F1

| Encoder | 1 frames | 5 frames | 15 frames | 30 frames |
| --- | ---: | ---: | ---: | ---: |
| scratch | 0.6622 ± 0.0028 | 0.6579 ± 0.0111 | 0.6351 ± 0.0081 | 0.5973 ± 0.0045 |
| tip_s3 | 0.6497 ± 0.0023 | 0.6694 ± 0.0100 | 0.6668 ± 0.0022 | 0.6455 ± 0.0182 |
| tip_s1b2048 | 0.6441 ± 0.0011 | 0.6420 ± 0.0072 | 0.6465 ± 0.0008 | 0.6280 ± 0.0069 |
| recon_tip_s1b2048 | 0.6434 ± 0.0080 | 0.6550 ± 0.0134 | 0.6492 ± 0.0013 | 0.6356 ± 0.0003 |

Δ vs scratch:

| Encoder | 1 frames | 5 frames | 15 frames | 30 frames |
| --- | ---: | ---: | ---: | ---: |
| tip_s3 | -0.0125 | +0.0116 | +0.0317 | +0.0482 |
| tip_s1b2048 | -0.0181 | -0.0159 | +0.0114 | +0.0307 |
| recon_tip_s1b2048 | -0.0188 | -0.0029 | +0.0141 | +0.0383 |

## grasp_prediction

### Last epoch — macro F1

| Encoder | 15 frames | 30 frames |
| --- | ---: | ---: |
| scratch | 0.8363 ± 0.0015 | 0.8438 ± 0.0030 |
| tip_s3 | 0.9218 ± 0.0016 | 0.9162 ± 0.0043 |
| tip_s1b2048 | 0.9139 ± 0.0081 | 0.9153 ± 0.0035 |
| recon_tip_s1b2048 | 0.9185 ± 0.0108 | 0.9195 ± 0.0004 |

Δ vs scratch:

| Encoder | 15 frames | 30 frames |
| --- | ---: | ---: |
| tip_s3 | +0.0855 | +0.0723 |
| tip_s1b2048 | +0.0776 | +0.0715 |
| recon_tip_s1b2048 | +0.0823 | +0.0756 |

### EpochAvg — macro F1

| Encoder | 15 frames | 30 frames |
| --- | ---: | ---: |
| scratch | 0.8297 ± 0.0001 | 0.8325 ± 0.0012 |
| tip_s3 | 0.9111 ± 0.0054 | 0.9057 ± 0.0011 |
| tip_s1b2048 | 0.9143 ± 0.0047 | 0.9065 ± 0.0006 |
| recon_tip_s1b2048 | 0.9067 ± 0.0066 | 0.8920 ± 0.0021 |

Δ vs scratch:

| Encoder | 15 frames | 30 frames |
| --- | ---: | ---: |
| tip_s3 | +0.0813 | +0.0732 |
| tip_s1b2048 | +0.0845 | +0.0741 |
| recon_tip_s1b2048 | +0.0770 | +0.0595 |

## Caveats

- Two seeds only; `±` is the spread between them, not a confidence interval.
- Window count falls sharply with frame length (slip: ~49k windows at 1 frame vs ~1.7k at 30), so frame length and sample count move together.
- With `require_uniform_label` off, the fraction of windows that straddle a slip boundary grows with the window: 3.0% at 3 frames, 17.1% at 15, 29.3% at 30. The label is taken from the last frame, so longer windows carry more label noise.
- ID protocol only; no leave-one-object-out.
