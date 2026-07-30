# Downstream results for the hp1 pretraining variants

Generated 2026-07-28 22:33

## Protocol

- ID protocol: episode-level 4-fold CV, seed 0, split seed 42
- Backbone LR 1e-4; probe LR 1e-4; batch 256
- Final `epoch-*.ckpt` of each pretraining run
- Metrics: balanced accuracy and macro F1 (last downstream epoch); `±` is fold-to-fold spread

## grasp_prediction

Scratch reference (3 seeds, same settings): **0.8467 ± 0.0058** macro F1.

| Pretraining variant | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048_prox | 100 | 0.9189 | 0.0093 | 0.9149 | 0.0053 | 0.9181 | OK |

Best: **dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048_prox** (macro F1 0.9149, bal acc 0.9189).

## slip_detection

Scratch reference (3 seeds, same settings): **0.6463 ± 0.0052** macro F1.

| Pretraining variant | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048_prox | 100 | 0.6800 | 0.0205 | 0.6443 | 0.0382 | 0.5759 | OK |

Best: **dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048_prox** (macro F1 0.6443, bal acc 0.6800).

## Caveats

- Single seed; `±` is fold-to-fold spread within each run.
- One backbone LR (1e-4); ID protocol only (no OOD).
- hp1 changes several hyperparameters at once, so a win here does not attribute to any single one.
