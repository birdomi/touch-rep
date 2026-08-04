# XYZ frame alignment: does matching HOI to BrainCo's fingertip frame help?

Generated 2026-07-31 10:40 — in progress, rows fill in as pretraining completes.

## The finding that started this

`BraincoSSLDataset` emitted raw wrist-local fingertip XYZ, while the BrainCo
*downstream* datasets ran `_align_xyz_to_npz` (on by default). So every BrainCo
SSL run so far pretrained on one fingertip frame and was evaluated on another.
The two frames are a cyclic axis permutation apart: HOI's long finger axis is y,
BrainCo's is z.

Direction chosen: convert **HOI into BrainCo's frame**
(`pseudo_force_tactile.align_xyz_to_brainco_frame`, the exact inverse of
`_align_xyz_to_npz`), so BrainCo keeps its raw FK output in pretraining,
downstream and inference alike, and only the HOI loader converts.

Alignment verified per finger, not just in aggregate: after conversion both
corpora show the lateral axis decreasing monotonically thumb → pinky, the same
sign on both hands, and index/middle longest on the length axis.

| | HOI (aligned) | BrainCo |
| --- | --- | --- |
| lateral y, thumb − pinky, left / right | +0.081 / +0.073 | +0.088 / +0.093 |
| length z, index / middle | 0.140 / 0.127 | 0.121 / 0.123 |
| palm-normal x, range | 0.053–0.068 | 0.028–0.041 |

The residual x gap is a human hand versus a robot hand, not a permutation error.

## Why per-axis normalization was dropped

The first attempt paired alignment with a per-axis z-score of the position
stream. Measured at pretraining epoch 200, seed 0, epoch-controlled against the
same checkpoint under the original pipeline:

| arm | grasp | slip |
| --- | ---: | ---: |
| HOI tip ep200, original pipeline | **0.9342** | **0.7671** |
| HOI tip ep200, aligned + z-scored | 0.9194 | 0.7412 |
| no pretraining, aligned + z-scored | 0.9214 | 0.7415 |

Normalization cost 1.5 points on grasp and 2.6 on slip, and erased the benefit
of pretraining entirely — the pretrained and untrained arms became
indistinguishable. It also flattens the ~1.8x palm-normal scale gap between the
two hands, which is real signal. All later runs use **alignment only**.

## Protocol

| | |
| --- | --- |
| grasp | `task/xyznorm/grasp_temporal_w15_cls_d4_fe4`, 4-fold CV |
| slip | `task/xyznorm/slip_temporal_w15_cls_d4_fe4_v3` (`slip_data_v3`), 3-fold CV |
| jointonly arm | `task/xyznorm/{grasp,slip}_jointonly` (`input_streams: pos`) |
| Overrides | `task.encoder_lr=1e-4`, `++split_seed=42`, 100 downstream epochs |
| Seeds | 0, 1 — `±` is the spread across the two |
| Pretraining | 200 epochs, lr 4e-4, warmup 10 (gentle: 5e-6, warmup 0) |
| Metric | balanced accuracy |

## Results — balanced accuracy

| # | HOI | BrainCo | Model | grasp (Best) | slip (Best) |
| ---: | :---: | :---: | --- | ---: | ---: |
| 1 | ✗ | ✗ | no pretraining | **0.8602 ± 0.0060** | **0.7519 ± 0.0036** |
| 2 | ✓ | ✗ | HOI tip | _pending_ | _pending_ |
| 3 | ✓ | ✗ | HOI 42j | _pending_ | _pending_ |
| 4 | ✓ | ✗ | HOI jointonly | _pending_ | _pending_ |
| 5 | ✗ | ✓ | brainco-only SSL | _pending_ | _pending_ |
| 6 | ✓ | ✓ | hoi-init SSL | _pending_ | _pending_ |
| 7 | ✓ | ✓ | hoi-init gentle 5e-6 | _pending_ | _pending_ |

### Row 1 in full

| readout | grasp | slip |
| --- | ---: | ---: |
| Best Epoch | 0.8602 ± 0.0060 | 0.7519 ± 0.0036 |
| Last Epoch | 0.8460 ± 0.0060 | 0.7097 ± 0.0023 |
| Epoch Average | 0.8396 ± 0.0010 | 0.7072 ± 0.0052 |
| Macro F1 (Best) | 0.8579 | 0.7346 |

The no-pretraining arm is unaffected by alignment — it has no pretrained pos
embedding to mismatch — so this row also serves as a check on the pipeline
change. It lands at 0.8602 grasp against 0.8615 in the original pipeline
(3 seeds), and 0.7519 slip against 0.7519. Identical within noise, as expected.

## Reference: the original pipeline, epoch 1000 / 500

For comparison, from `results_brainco_teleop_pretraining_20260730_222500.md`:

| Model | grasp | slip |
| --- | ---: | ---: |
| no pretraining | 0.8615 | 0.7519 |
| HOI tip (ep500) | 0.9411 | 0.7649 |
| HOI 42j (ep500) | 0.9370 | 0.7539 |
| HOI jointonly (ep500) | 0.8914 | 0.5446 |
| brainco-only SSL (ep1000) | 0.8790 | 0.7368 |
| hoi-init SSL (ep1000) | 0.9008 | 0.7697 |
| hoi-init gentle 1e-5 (ep100) | 0.9462 | 0.7704 |

Not row-comparable: those used 500/1000 pretraining epochs against 200 here, and
the BrainCo SSL rows there carry the pretraining/downstream frame mismatch this
experiment removes. The epoch-controlled reference point is HOI tip at ep200
under the original pipeline: **0.9342 grasp / 0.7671 slip** (seed 0).

## Caveats

- 2 seeds; fold spread is several times the seed spread in earlier rounds, so
  treat gaps under ~0.01 as unresolved.
- Rows 2–4 load 128/131 encoder tensors (`signal_mean`, `signal_std` and
  `sensor_embed.proj.weight` are dropped — proximity-only checkpoints against a
  4-channel task) and relearn the force projection downstream.
- One downstream backbone lr (1e-4), ID protocol only, no OOD.
