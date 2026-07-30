# Frame-length sweep — slip detection & grasp prediction

Generated 2026-07-28 23:25. 48/48 runs OK.

## Protocol

- Episode-level 4-fold CV, seeds 0,1, split seed 42
- Probe `BraincoGraspRoPEProbe` (depth 2, 3 heads); backbone LR 1e-4, probe LR 1e-4
- **`balanced_sampling` off, `require_uniform_label` off** (windows may straddle a slip boundary; the label is the last frame's)
- slip: `input_window_frames` = `input_window_stride` = the frame count; grasp: `window_time` = frames / 100 Hz
- Encoders:
  - `tip_s3` — `experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip/2026.07.28-13-23/checkpoints/epoch-0100.ckpt`
  - `tip_s1b2048` — `experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048/2026.07.28-14-48/checkpoints/epoch-0100.ckpt`
  - `recon_tip_s1b2048` — `experiments/dinov2_recon_all_pseudo_force_tiny_rope_v2_hp1_tip_s1_b2048/2026.07.28-14-49/checkpoints/epoch-0100.ckpt`
- Metrics: macro F1 at the **last** downstream epoch and the **EpochAvg** over epochs 10/20/30/40/50. `±` is the seed-to-seed spread; per-run fold std is in the CSV.

## slip_detection

### Last epoch — macro F1

| Encoder | 1 frames | 5 frames | 15 frames | 30 frames |
| --- | ---: | ---: | ---: | ---: |
| scratch | 0.6641 ± 0.0004 | 0.6614 ± 0.0123 | 0.6473 ± 0.0030 | 0.6658 ± 0.0069 |
| tip_s3 | 0.6562 ± 0.0107 | 0.6657 ± 0.0129 | 0.6756 ± 0.0035 | 0.6600 ± 0.0211 |
| tip_s1b2048 | 0.6537 ± 0.0081 | 0.6718 ± 0.0078 | 0.6612 ± 0.0098 | 0.6510 ± 0.0131 |
| recon_tip_s1b2048 | 0.6458 ± 0.0004 | 0.6722 ± 0.0127 | 0.6606 ± 0.0043 | 0.6668 ± 0.0021 |

Δ vs scratch:

| Encoder | 1 frames | 5 frames | 15 frames | 30 frames |
| --- | ---: | ---: | ---: | ---: |
| tip_s3 | -0.0079 | +0.0043 | +0.0282 | -0.0058 |
| tip_s1b2048 | -0.0104 | +0.0104 | +0.0139 | -0.0148 |
| recon_tip_s1b2048 | -0.0183 | +0.0108 | +0.0133 | +0.0010 |

### EpochAvg — macro F1

| Encoder | 1 frames | 5 frames | 15 frames | 30 frames |
| --- | ---: | ---: | ---: | ---: |
| scratch | 0.6622 ± 0.0028 | 0.6579 ± 0.0111 | 0.6351 ± 0.0081 | 0.5973 ± 0.0045 |
| tip_s3 | 0.6497 ± 0.0023 | 0.6694 ± 0.0100 | 0.6668 ± 0.0022 | 0.6455 ± 0.0182 |
| tip_s1b2048 | 0.6485 ± 0.0015 | 0.6643 ± 0.0011 | 0.6496 ± 0.0049 | 0.6292 ± 0.0132 |
| recon_tip_s1b2048 | 0.6544 ± 0.0017 | 0.6704 ± 0.0061 | 0.6483 ± 0.0133 | 0.6456 ± 0.0105 |

Δ vs scratch:

| Encoder | 1 frames | 5 frames | 15 frames | 30 frames |
| --- | ---: | ---: | ---: | ---: |
| tip_s3 | -0.0125 | +0.0116 | +0.0317 | +0.0482 |
| tip_s1b2048 | -0.0136 | +0.0065 | +0.0145 | +0.0319 |
| recon_tip_s1b2048 | -0.0078 | +0.0125 | +0.0132 | +0.0483 |

## grasp_prediction

### Last epoch — macro F1

| Encoder | 15 frames | 30 frames |
| --- | ---: | ---: |
| scratch | 0.8363 ± 0.0015 | 0.8438 ± 0.0030 |
| tip_s3 | 0.9218 ± 0.0016 | 0.9162 ± 0.0043 |
| tip_s1b2048 | 0.9226 ± 0.0055 | 0.9273 ± 0.0033 |
| recon_tip_s1b2048 | 0.9107 ± 0.0016 | 0.9203 ± 0.0090 |

Δ vs scratch:

| Encoder | 15 frames | 30 frames |
| --- | ---: | ---: |
| tip_s3 | +0.0855 | +0.0723 |
| tip_s1b2048 | +0.0863 | +0.0834 |
| recon_tip_s1b2048 | +0.0745 | +0.0764 |

### EpochAvg — macro F1

| Encoder | 15 frames | 30 frames |
| --- | ---: | ---: |
| scratch | 0.8297 ± 0.0001 | 0.8325 ± 0.0012 |
| tip_s3 | 0.9111 ± 0.0054 | 0.9057 ± 0.0011 |
| tip_s1b2048 | 0.9142 ± 0.0006 | 0.9084 ± 0.0035 |
| recon_tip_s1b2048 | 0.9103 ± 0.0002 | 0.9005 ± 0.0027 |

Δ vs scratch:

| Encoder | 15 frames | 30 frames |
| --- | ---: | ---: |
| tip_s3 | +0.0813 | +0.0732 |
| tip_s1b2048 | +0.0844 | +0.0760 |
| recon_tip_s1b2048 | +0.0806 | +0.0680 |

## Caveats

- Two seeds only; `±` is the spread between them, not a confidence interval.
- Window count falls sharply with frame length (slip: ~49k windows at 1 frame vs ~1.7k at 30), so frame length and sample count move together.
- With `require_uniform_label` off, the fraction of windows that straddle a slip boundary grows with the window: 3.0% at 3 frames, 17.1% at 15, 29.3% at 30. The label is taken from the last frame, so longer windows carry more label noise.
- ID protocol only; no leave-one-object-out.
