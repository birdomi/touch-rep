# Embedding sensitivity: force vs fingertip xyz

- metric: mean cosine distance between embeddings of real frames and the same
  frames with ONE input stream perturbed (shuffled across the batch / zeroed).
- 2000 frames per source; frame sources as in the raw comparison.

## dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix/epoch-0010

| source | force_shuffle | xyz_shuffle | both_shuffle | force_zero | xyz_mean | ch0_shuffle | ch12_shuffle | ch3_shuffle | force share* |
|---|---|---|---|---|---|---|---|---|---|
| pretrain | 0.1808 | 0.1197 | 0.2902 | 0.3273 | 0.0678 | 0.0104 | 0.0414 | 0.1743 | 0.60 |
| slip | 0.1153 | 0.0007 | 0.1160 | 0.3450 | 0.0003 | 0.0213 | 0.0318 | 0.1437 | 0.99 |
| grasp | 0.1328 | 0.0497 | 0.1787 | 0.3223 | 0.0266 | 0.0081 | 0.0289 | 0.1247 | 0.73 |

## dinov2_all_pseudo_force_tiny_rope_v2_hp1/epoch-0100

| source | force_shuffle | xyz_shuffle | both_shuffle | force_zero | xyz_mean | ch0_shuffle | ch12_shuffle | ch3_shuffle | force share* |
|---|---|---|---|---|---|---|---|---|---|
| pretrain | 0.1369 | 0.1288 | 0.2589 | 0.2684 | 0.0815 | 0.0118 | 0.0127 | 0.1399 | 0.52 |
| slip | 0.1100 | 0.0005 | 0.1105 | 0.3026 | 0.0003 | 0.0102 | 0.0116 | 0.1195 | 1.00 |
| grasp | 0.0823 | 0.0541 | 0.1340 | 0.2436 | 0.0292 | 0.0090 | 0.0097 | 0.0819 | 0.60 |

*force share = force_shuffle / (force_shuffle + xyz_shuffle); 0.5 = equally sensitive, 1.0 = force-only.
