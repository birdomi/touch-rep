# HOI data scaling and seed variance, on the BrainCo-frame pipeline

Generated 2026-08-02 13:58

## Protocol

| | |
| --- | --- |
| Alignment | HOI converted to BrainCo's fingertip frame; no per-axis whitening |
| grasp | `task/xyznorm/grasp_*`, 4-fold CV |
| slip | `task/xyznorm/slip_*_v3` on `slip_data_v3`, 3-fold CV |
| Overrides | `task.encoder_lr=1e-4`, `++split_seed=42`, 100 downstream epochs |
| Pretraining | 200 epochs, lr 4e-4, batch 2048 (BrainCo arms: 512 / 1e-4) |
| Data scaling | constant 49,200 optimizer steps at every fraction |
| Aggregation | mean ± sd over pretraining seeds x downstream seeds |

## HOI data scaling -- Epoch Average

| HOI data | grasp | slip | runs |
| ---: | ---: | ---: | ---: |
| 1% | 0.9196 ± 0.0049 | 0.7177 ± 0.0071 | 6/6 |
| 2% | 0.9248 ± 0.0044 | 0.7109 ± 0.0074 | 6/6 |
| 5% | 0.9280 ± 0.0045 | 0.7118 ± 0.0124 | 6/6 |
| 10% | 0.9245 ± 0.0028 | 0.7214 ± 0.0140 | 6/6 |
| 20% | 0.9261 ± 0.0047 | 0.7259 ± 0.0102 | 6/6 |
| 50% | 0.9316 ± 0.0035 | 0.7173 ± 0.0097 | 6/6 |
| 100% (seed 43) | 0.9260 ± 0.0088 | 0.7333 ± 0.0051 | 2/2 |

## HOI data scaling -- Last Epoch

| HOI data | grasp | slip | runs |
| ---: | ---: | ---: | ---: |
| 1% | 0.9308 ± 0.0051 | 0.7298 ± 0.0113 | 6/6 |
| 2% | 0.9323 ± 0.0035 | 0.7167 ± 0.0173 | 6/6 |
| 5% | 0.9338 ± 0.0048 | 0.7130 ± 0.0095 | 6/6 |
| 10% | 0.9321 ± 0.0044 | 0.7241 ± 0.0134 | 6/6 |
| 20% | 0.9341 ± 0.0057 | 0.7309 ± 0.0107 | 6/6 |
| 50% | 0.9388 ± 0.0034 | 0.7262 ± 0.0117 | 6/6 |
| 100% (seed 43) | 0.9340 ± 0.0041 | 0.7350 ± 0.0028 | 2/2 |

## HOI data scaling -- Best Epoch

| HOI data | grasp | slip | runs |
| ---: | ---: | ---: | ---: |
| 1% | 0.9409 ± 0.0033 | 0.7706 ± 0.0080 | 6/6 |
| 2% | 0.9434 ± 0.0056 | 0.7574 ± 0.0050 | 6/6 |
| 5% | 0.9443 ± 0.0029 | 0.7548 ± 0.0128 | 6/6 |
| 10% | 0.9412 ± 0.0029 | 0.7701 ± 0.0088 | 6/6 |
| 20% | 0.9441 ± 0.0053 | 0.7673 ± 0.0064 | 6/6 |
| 50% | 0.9478 ± 0.0036 | 0.7651 ± 0.0101 | 6/6 |
| 100% (seed 43) | 0.9414 ± 0.0049 | 0.7807 ± 0.0020 | 2/2 |

## Arms at pretraining seed 43 -- Epoch Average

| Model | grasp | slip | runs |
| --- | ---: | ---: | ---: |
| HOI tip | 0.9260 ± 0.0088 | 0.7333 ± 0.0051 | 2/2 |
| HOI jointonly | 0.8668 ± 0.0045 | 0.5134 ± 0.0088 | 2/2 |
| brainco-only SSL | 0.8637 ± 0.0025 | 0.6849 ± 0.0139 | 2/2 |
| brainco-only jointonly | 0.8339 ± 0.0014 | 0.5116 ± 0.0039 | 2/2 |
| hoi-init SSL | 0.9334 ± 0.0053 | 0.7173 ± 0.0033 | 2/2 |
| hoi-init gentle | 0.9316 ± 0.0028 | 0.7294 ± 0.0071 | 2/2 |
| frame-w1 tip | 0.9143 ± 0.0065 | 0.5723 ± 0.0105 | 2/2 |
| frame-w1 42j | 0.9213 ± 0.0017 | 0.5919 ± 0.0162 | 2/2 |

## Arms at pretraining seed 43 -- Last Epoch

| Model | grasp | slip | runs |
| --- | ---: | ---: | ---: |
| HOI tip | 0.9340 ± 0.0041 | 0.7350 ± 0.0028 | 2/2 |
| HOI jointonly | 0.8770 ± 0.0042 | 0.5169 ± 0.0157 | 2/2 |
| brainco-only SSL | 0.8661 ± 0.0061 | 0.6980 ± 0.0065 | 2/2 |
| brainco-only jointonly | 0.8387 ± 0.0024 | 0.5277 ± 0.0004 | 2/2 |
| hoi-init SSL | 0.9360 ± 0.0049 | 0.7181 ± 0.0104 | 2/2 |
| hoi-init gentle | 0.9387 ± 0.0057 | 0.7295 ± 0.0251 | 2/2 |
| frame-w1 tip | 0.9156 ± 0.0061 | 0.5766 ± 0.0054 | 2/2 |
| frame-w1 42j | 0.9247 ± 0.0016 | 0.5824 ± 0.0158 | 2/2 |

## Arms at pretraining seed 43 -- Best Epoch

| Model | grasp | slip | runs |
| --- | ---: | ---: | ---: |
| HOI tip | 0.9414 ± 0.0049 | 0.7807 ± 0.0020 | 2/2 |
| HOI jointonly | 0.8916 ± 0.0004 | 0.5639 ± 0.0000 | 2/2 |
| brainco-only SSL | 0.8880 ± 0.0037 | 0.7107 ± 0.0141 | 2/2 |
| brainco-only jointonly | 0.8767 ± 0.0011 | 0.5570 ± 0.0000 | 2/2 |
| hoi-init SSL | 0.9418 ± 0.0024 | 0.7421 ± 0.0079 | 2/2 |
| hoi-init gentle | 0.9441 ± 0.0025 | 0.7669 ± 0.0003 | 2/2 |
| frame-w1 tip | 0.9341 ± 0.0012 | 0.6048 ± 0.0050 | 2/2 |
| frame-w1 42j | 0.9415 ± 0.0022 | 0.6196 ± 0.0112 | 2/2 |

## Caveats

- `±` spans pretraining seeds and downstream seeds together, so it mixes both sources of variance; per-fold spread is larger still.
- The seed-42 numbers for the 8-row table live in `results_xyz_alignment_20260731.md`; compare against those for the pretraining-seed effect.
- The frame-w1 arms use the frame-wise task stack (`task/xyznorm/*_framew1`), not the temporal former, so they are not directly comparable to the other rows.
