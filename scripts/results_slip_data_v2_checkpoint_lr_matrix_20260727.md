# slip_data_v2: Checkpoint × Backbone LR Matrix

## Protocol

- Data: `dataset/brainco/downstream/slip_data_v2`
- Classes/episodes: box 26, doll 25, evacase 25, plastic 21; total 97
- Dataset manifest SHA-256: `6f6d9eed51637355be7ab12ed5668b584dcb62054cf5342327d533aa88221241`
- Episode-level 4-fold CV; seed 0; split seed 42
- 50 epochs; probe LR `1e-4`; backbone LR: `1e-4` or `1e-5`
- Temporal input: three 3-frame chunks; transition exclusion: 0 frames
- Eight fresh downstream runs executed concurrently on GPUs 2 and 6
- Common encoder structure: mask token enabled, drop path 0.3, RoPE hand offset 10

## Models

- Scratch: random initialization
- V2 e10: `dinov2_all_pseudo_force_tiny_rope_temporal3_v2/.../epoch-0010.ckpt`
- Base e100: `dinov2_all_pseudo_force_tiny_rope_temporal3/.../epoch-0100.ckpt`
- Local e8: fixed `last-epoch0008-matrix-snapshot.ckpt`

## Aggregate Results

Values are mean ± population standard deviation across four folds.

| Model | Backbone LR | Last Acc | Last F1 | Best Acc | Best F1 | EpochAvg Acc | EpochAvg F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch | 1e-4 | 0.6313 ± 0.0051 | 0.5352 ± 0.0306 | 0.6858 ± 0.0400 | 0.2860 ± 0.2865 | 0.6067 ± 0.0263 | 0.5192 ± 0.0329 |
| Scratch | 1e-5 | 0.5968 ± 0.0251 | 0.5216 ± 0.0392 | 0.6774 ± 0.0533 | 0.1158 ± 0.2006 | 0.5928 ± 0.0230 | 0.5175 ± 0.0305 |
| V2 e10 | 1e-4 | 0.6309 ± 0.0262 | 0.5173 ± 0.0429 | 0.6547 ± 0.0292 | 0.5282 ± 0.0322 | 0.6215 ± 0.0242 | 0.5334 ± 0.0343 |
| V2 e10 | 1e-5 | 0.6398 ± 0.0248 | 0.5382 ± 0.0464 | 0.6558 ± 0.0227 | 0.5413 ± 0.0473 | 0.6262 ± 0.0249 | 0.5321 ± 0.0416 |
| Base e100 | 1e-4 | 0.6638 ± 0.0175 | 0.5577 ± 0.0350 | 0.6900 ± 0.0198 | 0.3150 ± 0.2663 | 0.6615 ± 0.0167 | 0.5587 ± 0.0332 |
| Base e100 | 1e-5 | 0.6704 ± 0.0289 | 0.5760 ± 0.0133 | 0.6882 ± 0.0262 | 0.4586 ± 0.2235 | 0.6559 ± 0.0289 | 0.5685 ± 0.0093 |
| Local e8 | 1e-4 | 0.6576 ± 0.0192 | 0.5446 ± 0.0562 | 0.6940 ± 0.0305 | 0.2912 ± 0.2927 | 0.6510 ± 0.0260 | 0.5391 ± 0.0402 |
| Local e8 | 1e-5 | 0.6471 ± 0.0252 | 0.5573 ± 0.0356 | 0.6940 ± 0.0447 | 0.3054 ± 0.3055 | 0.6428 ± 0.0388 | 0.5627 ± 0.0367 |

## Backbone LR Effect

EpochAvg change for `1e-5 - 1e-4`:

| Model | Accuracy | F1 |
| --- | ---: | ---: |
| Scratch | -0.0139 | -0.0017 |
| V2 e10 | +0.0047 | -0.0013 |
| Base e100 | -0.0056 | +0.0098 |
| Local e8 | -0.0082 | +0.0236 |

Unlike the prior move dataset, LR `1e-5` does not uniformly win. For accuracy, Base e100 at `1e-4` has the best EpochAvg. For F1 and Last metrics, Base e100 at `1e-5` is best.

The `Best` checkpoint is selected by validation accuracy. Several runs select class-collapsed checkpoints with low F1, so Last and EpochAvg are more reliable for this comparison.

## Output Directories

All eight directories are under `experiments/brainco_xyz_slip_detection/` with timestamp `2026.07.27_15-19` and names:

- `v2data_scratch_blr1e4`, `v2data_scratch_blr1e5`
- `v2data_v2e10_blr1e4`, `v2data_v2e10_blr1e5`
- `v2data_basee100_blr1e4`, `v2data_basee100_blr1e5`
- `v2data_locale8_blr1e4`, `v2data_locale8_blr1e5`

All eight processes exited with code 0. Each output contains the expected four-fold Last, Best, and EpochAvg artifacts.
