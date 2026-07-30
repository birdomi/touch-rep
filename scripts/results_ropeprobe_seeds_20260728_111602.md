# BraincoGraspRoPEProbe: scratch vs `rope` pretraining, 3 seeds

Generated 2026-07-28 11:46. 12/12 runs OK.

## Protocol

- Probe: `BraincoGraspRoPEProbe`, depth 2, 3 heads (causal RoPE attention, classifies from the last time step) — replaces `MeanPoolProbe`
- Episode-level 4-fold CV, seeds 0,1,2, split seed 42
- Backbone LR 1e-4, probe LR 1e-4, 50 epochs, train batch 256
- Encoders: `scratch` (random init) vs `rope` (`experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1/2026.07.27-22-14/checkpoints/epoch-0100.ckpt`)
- Grasp prediction: 30-frame windows (data-config default). Slip detection: 3-frame windows.
- Metrics: **balanced accuracy** (mean per-class recall) and **macro F1**, last epoch.
- `±` in the summary tables is the **seed-to-seed** std; per-run fold std is in the CSV.

## grasp_prediction

| Encoder | Bal Acc | Macro F1 | F1 (bin) | n seeds |
| --- | ---: | ---: | ---: | ---: |
| scratch | 0.8501 ± 0.0061 | 0.8475 ± 0.0067 | 0.8555 ± 0.0068 | 3 |
| rope | 0.9272 ± 0.0054 | 0.9262 ± 0.0061 | 0.9295 ± 0.0077 | 3 |

Δ macro F1 (`rope` − scratch): **+0.0787** — larger than the scratch seed spread (±0.0067).

### Per-seed (last epoch)

| Encoder | Seed | Bal Acc | Macro F1 | fold std (macro F1) | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| scratch | 0 | 0.8460 | 0.8417 | 0.0193 | OK |
| scratch | 1 | 0.8472 | 0.8460 | 0.0060 | OK |
| scratch | 2 | 0.8571 | 0.8549 | 0.0200 | OK |
| rope | 0 | 0.9304 | 0.9311 | 0.0116 | OK |
| rope | 1 | 0.9302 | 0.9282 | 0.0207 | OK |
| rope | 2 | 0.9209 | 0.9193 | 0.0291 | OK |

## slip_detection

| Encoder | Bal Acc | Macro F1 | F1 (bin) | n seeds |
| --- | ---: | ---: | ---: | ---: |
| scratch | 0.6883 ± 0.0087 | 0.6583 ± 0.0096 | 0.5859 ± 0.0076 | 3 |
| rope | 0.6847 ± 0.0126 | 0.6452 ± 0.0117 | 0.5864 ± 0.0132 | 3 |

Δ macro F1 (`rope` − scratch): **-0.0131** — larger than the scratch seed spread (±0.0096).

### Per-seed (last epoch)

| Encoder | Seed | Bal Acc | Macro F1 | fold std (macro F1) | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| scratch | 0 | 0.6981 | 0.6677 | 0.0339 | OK |
| scratch | 1 | 0.6814 | 0.6486 | 0.0330 | OK |
| scratch | 2 | 0.6853 | 0.6585 | 0.0344 | OK |
| rope | 0 | 0.6992 | 0.6578 | 0.0351 | OK |
| rope | 1 | 0.6788 | 0.6430 | 0.0248 | OK |
| rope | 2 | 0.6761 | 0.6347 | 0.0214 | OK |

## Caveats

- One backbone LR and one probe LR; neither was swept for the larger probe (890k params vs 386 for `MeanPoolProbe`).
- ID protocol only (episode-level K-fold over all objects); no leave-one-object-out.
- Only the `rope` checkpoint is compared; `temp3` and `prediction_local` are not in this run.
dd