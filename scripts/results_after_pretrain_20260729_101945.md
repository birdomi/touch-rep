# Downstream results for the hp1 pretraining variants

Generated 2026-07-29 10:29

## Protocol

- ID protocol: episode-level 4-fold CV, seed 0, split seed 42
- Backbone LR 1e-4; probe LR 1e-4; batch 256
- Final `epoch-*.ckpt` of each pretraining run
- Metrics: balanced accuracy and macro F1 (last downstream epoch); `±` is fold-to-fold spread

## grasp_prediction

Scratch reference (3 seeds, same settings): **0.8467 ± 0.0058** macro F1.

| Pretraining variant | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048_ep500 | 330 | 0.9309 | 0.0076 | 0.9299 | 0.0069 | 0.9332 | OK |
| recon_rope_v2_hp1_tip_s1_b2048_ep500 | 370 | 0.9374 | 0.0095 | 0.9368 | 0.0094 | 0.9383 | OK |

Best: **dinov2_recon_all_pseudo_force_tiny_rope_v2_hp1_tip_s1_b2048_ep500** (macro F1 0.9368, bal acc 0.9374).

## slip_detection

Scratch reference (3 seeds, same settings): **0.6463 ± 0.0052** macro F1.

| Pretraining variant | Epoch | Bal Acc | ± | F1 macro | ± | F1 (bin) | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all_pseudo_force_tiny_rope_v2_hp1_ibotfix_tip_s1_b2048_ep500 | 330 | n/a | n/a | n/a | n/a | n/a | FAILED |
| recon_rope_v2_hp1_tip_s1_b2048_ep500 | 370 | n/a | n/a | n/a | n/a | n/a | FAILED |
## Caveats

- Single seed; `±` is fold-to-fold spread within each run.
- One backbone LR (1e-4); ID protocol only (no OOD).
- hp1 changes several hyperparameters at once, so a win here does not attribute to any single one.
