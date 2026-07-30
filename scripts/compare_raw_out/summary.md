# Raw data comparison: pretraining vs downstream

- pretrain: `pretraining_dataset/brainco` — 12 episodes, 1973 frames (stride 10)
- downstream: `dataset/brainco/downstream/grasp_prediction` — 30 episodes, 1694 frames (stride 1)
- fingertip xyz: wrist(base_link)-relative, via `compute_fk`
- NN matching: per-sensor KDTree on xyz, k=5; 'close pair' threshold = 5.0 mm

## Raw tactile channel stats (all frames x sensors)

| channel | source | valid% | mean | std | p50 | p95 | max |
|---|---|---|---|---|---|---|---|
| ch0 normal_force | pretrain | 100.0 | 16.4 | 62.7 | 0.0 | 85.0 | 803.0 |
| ch0 normal_force | downstream | 100.0 | 23.9 | 74.8 | 0.0 | 137.0 | 944.0 |
| ch1 tangential_force | pretrain | 100.0 | 26.3 | 122.4 | 0.0 | 116.5 | 1771.0 |
| ch1 tangential_force | downstream | 100.0 | 33.2 | 134.6 | 0.0 | 155.0 | 2912.0 |
| ch2 direction | pretrain | 19.4 | 155.4 | 111.0 | 123.0 | 350.0 | 359.0 |
| ch2 direction | downstream | 33.7 | 126.1 | 116.3 | 113.0 | 321.0 | 356.0 |
| ch3 proximity | pretrain | 100.0 | 157506.6 | 532605.7 | 0.0 | 878509.9 | 8109079.0 |
| ch3 proximity | downstream | 100.0 | 128303.9 | 542185.9 | 0.0 | 593469.6 | 11305290.0 |

## Fingertip-xyz NN distance (downstream -> nearest pretrain, same sensor)

| sensor | p50 (mm) | p95 (mm) | max (mm) | close pairs (<thr) |
|---|---|---|---|---|
| L_thumb | 0.27 | 0.79 | 0.97 | 1694/1694 |
| L_index | 0.00 | 0.09 | 0.09 | 1694/1694 |
| L_middle | 0.00 | 0.10 | 0.19 | 1694/1694 |
| L_ring | 0.00 | 0.09 | 0.25 | 1694/1694 |
| L_pinky | 0.00 | 0.08 | 0.15 | 1694/1694 |
| R_thumb | 0.43 | 1.29 | 5.11 | 1692/1694 |
| R_index | 0.00 | 0.16 | 0.40 | 1694/1694 |
| R_middle | 0.00 | 0.18 | 0.48 | 1694/1694 |
| R_ring | 0.00 | 0.25 | 1.31 | 1694/1694 |
| R_pinky | 0.00 | 0.14 | 0.42 | 1694/1694 |

## Matched-pair tactile diff (close pairs only, 16938 pairs)

downstream value vs mean of its k nearest pretraining neighbours:

| channel | mean abs diff | p50 abs diff | p95 abs diff | corr |
|---|---|---|---|---|
| ch0 normal_force | 29.9 | 2.0 | 147.1 | 0.109 |
| ch1 tangential_force | 44.1 | 1.8 | 186.2 | 0.074 |
| ch2 direction | 134.3 | 107.4 | 312.0 | -0.164 |
| ch3 proximity | 152849.4 | 29859.6 | 592659.2 | 0.052 |
