# Embedding-space comparison (tactile encoder)

- encoder: `experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1/2026.07.27-22-14/checkpoints/epoch-0100.ckpt` (teacher backbone, reg-token mean, L2-normalized)
- pretrain: 1973 frames | slip: 6886 frames (slip label ratio 0.38) | grasp: 1694 frames

## Centroid cosine similarity

- pretrain-slip: 0.8725
- pretrain-grasp: 0.9479
- slip-grasp: 0.7898

## NN cosine distance to pretrain set

- pretrain -> pretrain (LOO, baseline): p50=0.0034, p95=0.0321, max=0.0768
- slip -> pretrain: p50=0.0538, p95=0.0775, max=0.1083
- grasp -> pretrain: p50=0.0414, p95=0.0700, max=0.0933

## kNN domain purity vs pretrain (0.5 = indistinguishable, 1.0 = separated)

- slip vs pretrain: 0.997
- grasp vs pretrain: 0.973

## Slip-label separability inside slip embeddings (LOO kNN, k=10)

- LOO (within-episode leakage possible): balanced acc 0.855 (non-slip recall 0.920, slip recall 0.791)
- cross-episode (own episode excluded): balanced acc 0.593 (non-slip recall 0.798, slip recall 0.387)
