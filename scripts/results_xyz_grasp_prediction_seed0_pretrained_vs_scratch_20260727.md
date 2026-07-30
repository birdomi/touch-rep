# XYZ Grasp Prediction — Pretrained vs Scratch (seed 0)

- Protocol: ID, episode-level 4-fold (`split_seed=42`), `--all_split --num_folds 4`
- Seed: 0 only (sweep stopped after the seed-0 pair)
- Encoder checkpoint: `checkpoints/dinov2_xyz/epoch-0100.ckpt`
- `max_values = [1000, 1000, 1000, 100000]` (changed 2026-07-27; earlier result files used `[25000, 25000, 25000, 500000]`)
- Single GPU (`CUDA_VISIBLE_DEVICES=1`), 50 epochs, `val_average_epochs = [10, 20, 30, 40, 50]`
- Data: 590 episodes = 590 windows (31 frames/episode, 30-frame window → 1 window each); 443 train / 147 val per fold
- Logs: `scripts/logs/xyz_grasp_prediction_id_multiseed_20260727_161623/`

## Mean over 4 folds (± fold std)

| Metric | Pretrained | Scratch | Δ |
| --- | ---: | ---: | ---: |
| Last Acc | 0.8831 ± 0.0293 | 0.8406 ± 0.0177 | +0.0425 |
| Last F1 | 0.8833 ± 0.0390 | 0.8382 ± 0.0207 | +0.0451 |
| Best Acc | 0.9067 ± 0.0179 | 0.8508 ± 0.0134 | +0.0559 |
| Best F1 | 0.9103 ± 0.0145 | 0.8457 ± 0.0143 | +0.0646 |
| EpochAvg Acc | 0.8633 ± 0.0209 | 0.8298 ± 0.0152 | +0.0335 |
| EpochAvg F1 | 0.8651 ± 0.0270 | 0.8269 ± 0.0206 | +0.0382 |

## Per-fold accuracy

| Fold | Pretrained Last | Scratch Last | Δ | Pretrained EpochAvg | Scratch EpochAvg | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8367 | 0.8163 | +0.0204 | 0.8272 | 0.8054 | +0.0218 |
| 1 | 0.9048 | 0.8435 | +0.0613 | 0.8748 | 0.8354 | +0.0394 |
| 2 | 0.9116 | 0.8367 | +0.0749 | 0.8762 | 0.8313 | +0.0449 |
| 3 | 0.8792 | 0.8658 | +0.0134 | 0.8752 | 0.8470 | +0.0282 |

Pretrained wins every fold on both metrics.

## Caveats

- **Single seed.** The ± above is fold-to-fold spread, not seed spread. Fold std (~0.02-0.03) is comparable to the gap on the weaker folds (0 and 3).
- **Learning rate is confounded with pretraining.** `dinov2_all_rope` uses `encoder_lr=1e-6`, `dinov2_all_rope_scratch` uses `encoder_lr=1e-4` (`task_lr=1e-4` for both). A randomly initialized encoder at 1e-6 would not train, so the configs cannot be matched trivially — but the comparison measures "pretrained + low encoder LR" vs "scratch + high encoder LR", not pretraining alone.
- **Not comparable to earlier result files** in `scripts/`, which predate the `max_values` change.
