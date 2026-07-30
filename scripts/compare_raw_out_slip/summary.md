# Raw data comparison: pretraining vs downstream

- pretrain: `pretraining_dataset/brainco` — 12 episodes, 1973 frames (stride 10)
- downstream: `dataset/brainco/downstream/slip_data_v2` — 24 episodes, 6886 frames (stride 2)
- fingertip xyz: wrist(base_link)-relative, via `compute_fk`
- NN matching: per-sensor KDTree on xyz, k=5; 'close pair' threshold = 5.0 mm

## Raw tactile channel stats (all frames x sensors)

| channel | source | valid% | mean | std | p50 | p95 | max |
|---|---|---|---|---|---|---|---|
| ch0 normal_force | pretrain | 100.0 | 16.4 | 62.7 | 0.0 | 85.0 | 803.0 |
| ch0 normal_force | downstream | 100.0 | 29.4 | 82.1 | 0.0 | 167.0 | 1163.0 |
| ch1 tangential_force | pretrain | 100.0 | 26.3 | 122.4 | 0.0 | 116.5 | 1771.0 |
| ch1 tangential_force | downstream | 100.0 | 36.4 | 141.5 | 0.0 | 161.0 | 2248.0 |
| ch2 direction | pretrain | 19.4 | 155.4 | 111.0 | 123.0 | 350.0 | 359.0 |
| ch2 direction | downstream | 27.3 | 166.8 | 82.4 | 173.0 | 284.0 | 359.0 |
| ch3 proximity | pretrain | 100.0 | 157506.6 | 532605.7 | 0.0 | 878509.9 | 8109079.0 |
| ch3 proximity | downstream | 100.0 | 278886.1 | 578123.4 | 0.0 | 1729855.2 | 4300926.0 |

## Fingertip-xyz NN distance (downstream -> nearest pretrain, same sensor)

| sensor | p50 (mm) | p95 (mm) | max (mm) | close pairs (<thr) |
|---|---|---|---|---|
| L_thumb | 24.46 | 24.61 | 24.69 | 0/6886 |
| L_index | 0.00 | 0.00 | 0.09 | 6886/6886 |
| L_middle | 0.00 | 0.00 | 0.10 | 6886/6886 |
| L_ring | 0.00 | 0.00 | 0.00 | 6886/6886 |
| L_pinky | 0.00 | 0.00 | 0.00 | 6886/6886 |
| R_thumb | 27.60 | 28.13 | 28.13 | 0/6886 |
| R_index | 0.00 | 0.00 | 0.00 | 6886/6886 |
| R_middle | 0.00 | 0.00 | 0.00 | 6886/6886 |
| R_ring | 0.00 | 0.00 | 0.00 | 6886/6886 |
| R_pinky | 0.00 | 0.00 | 0.00 | 6886/6886 |

## Matched-pair tactile diff (close pairs only, 55088 pairs)

downstream value vs mean of its k nearest pretraining neighbours:

| channel | mean abs diff | p50 abs diff | p95 abs diff | corr |
|---|---|---|---|---|
| ch0 normal_force | 39.8 | 4.0 | 207.8 | -0.068 |
| ch1 tangential_force | 48.8 | 3.0 | 206.0 | -0.036 |
| ch2 direction | 94.9 | 82.0 | 204.0 | -0.314 |
| ch3 proximity | 352196.7 | 73858.4 | 1897358.0 | -0.137 |
