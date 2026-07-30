# Grasp Prediction: Checkpoint Matrix at Backbone LR 1e-4

## Protocol

- Data: `dataset/brainco/downstream/grasp_prediction_0611`
- Labels: box/case/driver/eraser/tumbler, success and failure
- Total: 590 episodes
- Dataset manifest SHA-256: `28d0c6d7fe7d67bdfb84e3902b51346c73010582d53d1dc9fa42992e7654d923`
- Episode-level 4-fold CV; seed 0; split seed 42
- 50 epochs; backbone LR `1e-4`; probe LR `1e-4`
- 30-frame grasp window, encoded as ten consecutive 3-frame chunks
- Four fresh downstream runs executed concurrently on GPUs 2 and 6

## Models

- Scratch: random initialization
- V2 e10: `dinov2_all_pseudo_force_tiny_rope_temporal3_v2/.../epoch-0010.ckpt`
- Base e100: `dinov2_all_pseudo_force_tiny_rope_temporal3/.../epoch-0100.ckpt`
- Local e8: fixed `last-epoch0008-matrix-snapshot.ckpt`

## Aggregate Results

Values are mean ± population standard deviation across four folds.

| Model | Last Acc | Last F1 | Best Acc | Best F1 | EpochAvg Acc | EpochAvg F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch | 0.8304 ± 0.0135 | 0.8259 ± 0.0188 | 0.8491 ± 0.0154 | 0.8446 ± 0.0190 | 0.8270 ± 0.0190 | 0.8243 ± 0.0182 |
| V2 e10 | 0.9152 ± 0.0197 | 0.9161 ± 0.0209 | 0.9322 ± 0.0144 | 0.9334 ± 0.0169 | 0.8871 ± 0.0189 | 0.8869 ± 0.0193 |
| Base e100 | **0.9254 ± 0.0096** | **0.9264 ± 0.0117** | **0.9356 ± 0.0149** | **0.9359 ± 0.0167** | **0.9014 ± 0.0179** | **0.9034 ± 0.0204** |
| Local e8 | **0.9254 ± 0.0110** | 0.9263 ± 0.0107 | 0.9288 ± 0.0115 | 0.9294 ± 0.0115 | 0.8925 ± 0.0148 | 0.8950 ± 0.0160 |

## Summary

- Base e100 is the overall winner for Last, Best, and EpochAvg.
- Local e8 ties Base e100 on Last accuracy and trails Last F1 by 0.0001.
- All pretrained encoders strongly outperform Scratch.
- Last improvement over Scratch: Base e100 `+0.0950 Acc / +0.1005 F1`; Local e8 `+0.0950 / +0.1004`; V2 e10 `+0.0848 / +0.0902`.

## Output Directories

All directories are under `experiments/brainco_xyz_grasp_prediction/` with timestamp `2026.07.27_15-53`:

- `grasp2_scratch_blr1e4`
- `grasp2_v2e10_blr1e4`
- `grasp2_basee100_blr1e4`
- `grasp2_locale8_blr1e4`

All four processes exited with code 0. Each output contains the expected four-fold Last, Best, and EpochAvg artifacts.
