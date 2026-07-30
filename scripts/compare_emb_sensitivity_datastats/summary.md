# Embedding sensitivity: force vs fingertip xyz

- metric: mean cosine distance between embeddings of real frames and the same
  frames with ONE input stream perturbed (shuffled across the batch / zeroed).
- 2000 frames per source; frame sources as in the raw comparison.

## dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix/epoch-0010

| source | force_shuffle | xyz_shuffle | both_shuffle | force_zero | xyz_mean | ch0_shuffle | ch12_shuffle | ch3_shuffle | force share* |
|---|---|---|---|---|---|---|---|---|---|
| pretrain | 0.1427 | 0.1338 | 0.2631 | 0.1027 | 0.0795 | 0.0425 | 0.0649 | 0.0946 | 0.52 |
| slip | 0.2086 | 0.0008 | 0.2093 | 0.1771 | 0.0004 | 0.0913 | 0.0748 | 0.1500 | 1.00 |
| grasp | 0.1087 | 0.0587 | 0.1631 | 0.0944 | 0.0312 | 0.0387 | 0.0515 | 0.0515 | 0.65 |

## dinov2_all_pseudo_force_tiny_rope_v2_hp1/epoch-0100

| source | force_shuffle | xyz_shuffle | both_shuffle | force_zero | xyz_mean | ch0_shuffle | ch12_shuffle | ch3_shuffle | force share* |
|---|---|---|---|---|---|---|---|---|---|
| pretrain | 0.1144 | 0.1211 | 0.2257 | 0.0913 | 0.0751 | 0.0611 | 0.0084 | 0.0973 | 0.49 |
| slip | 0.1563 | 0.0005 | 0.1567 | 0.1429 | 0.0003 | 0.0786 | 0.0064 | 0.1743 | 1.00 |
| grasp | 0.0770 | 0.0401 | 0.1147 | 0.0728 | 0.0216 | 0.0614 | 0.0032 | 0.0223 | 0.66 |

*force share = force_shuffle / (force_shuffle + xyz_shuffle); 0.5 = equally sensitive, 1.0 = force-only.
