# Pretraining checkpoint survey — grasp prediction & slip detection

All runs: seed 0, episode-level 4-fold CV, split seed 42, 50 downstream epochs.
Probe `BraincoGraspRoPEProbe` (depth 2, 3 heads), backbone LR 1e-4, probe LR 1e-4,
train batch 256. Grasp uses 30-frame windows, slip uses 3-frame windows.

**Two metrics are reported side by side** (both macro F1):

- **Last** — the final downstream epoch (50). No validation-based selection, so no
  optimistic bias.
- **EpochAvg** — mean over downstream epochs 10/20/30/40/50 (`val_average_epochs`).
  Less sensitive to where the last epoch happens to land.

`Best` (per-fold best validation epoch) is in the logs but is not reported here: it
selects on the validation split and runs 0.01–0.04 optimistic.

## grasp_prediction — macro F1

| Encoder | Pretrain epoch | Last | EpochAvg |
| --- | ---: | ---: | ---: |
| scratch | — | 0.8400 | 0.8246 |
| `rope` (pre-iBOT-fix) | 100 | 0.9311 | **0.9111** |
| `ibotfix` (42 joints) | 10 | 0.8753 | 0.8523 |
| `recon` (42 joints) | 10 | 0.8759 | 0.8509 |
| `recon` | 20 | 0.8920 | 0.8667 |
| `recon` | 30 | 0.8978 | 0.8995 |
| `recon` | 40 | 0.9180 | 0.8987 |
| `recon` | 50 | 0.9194 | 0.9063 |
| `tip` (10 fingertips) | 10 | 0.8894 | 0.8599 |
| `tip` | 20 | 0.9231 | 0.8972 |
| `tip` | 30 | 0.9145 | 0.9072 |
| **`tip`** | **50** | **0.9263** | 0.9077 |
| `tip_s1_b2048` | 10 | 0.8453 | 0.8452 |
| `recon_tip_s1_b2048` | 10 | 0.8602 | 0.8426 |

The two metrics agree on ordering except at the top: **Last picks `tip` e50 (0.9263),
EpochAvg picks `rope` e100 (0.9111)**. `tip` e30 and e50 have essentially identical
EpochAvg (0.9072 / 0.9077), so `tip` is saturated by epoch 30.

## slip_detection — macro F1

| Encoder | Pretrain epoch | Last | EpochAvg |
| --- | ---: | ---: | ---: |
| scratch | — | 0.6406 | 0.6501 |
| `rope` (pre-iBOT-fix) | 100 | 0.6578 | 0.6606 |
| `ibotfix` | 10 | 0.6627 | 0.6540 |
| `recon` | 10 | 0.6565 | 0.6554 |
| `recon` | 20 | 0.6520 | 0.6501 |
| `recon` | 30 | 0.6493 | 0.6543 |
| `recon` | 40 | 0.6177 | 0.6426 |
| `recon` | 50 | 0.6587 | 0.6644 |
| `tip` | 10 | 0.6673 | 0.6545 |
| **`tip`** | **20** | **0.6860** | 0.6742 |
| `tip` | 30 | 0.6536 | 0.6538 |
| `tip` | 50 | 0.6504 | 0.6552 |
| `tip_s1_b2048` | 10 | 0.6580 | 0.6620 |
| **`recon_tip_s1_b2048`** | 10 | 0.6785 | **0.6793** |

Here the two metrics disagree more: **Last picks `tip` e20 (0.6860), EpochAvg picks
`recon_tip_s1_b2048` e10 (0.6793)**. Every pretrained EpochAvg except `recon` e20/e40
clears scratch (0.6501), whereas on Last several fall below it — the Last column on slip
is noisy enough to invert the scratch comparison.

## Findings

**1. Fingertip-only pretraining is the clearest win on grasp.** At a matched epoch 10
with the same post-iBOT-fix code, `tip` beats `ibotfix` by +0.0141 Last / +0.0076
EpochAvg. Restricting the pretraining token set to the 10 fingertips the downstream
actually senses — rather than 42 hand joints, 32 of which the downstream never uses — is
worth more than any objective change tested here.

**2. The reconstruction objective adds nothing at stride 3.** `recon` vs `ibotfix` at
epoch 10: +0.0006 Last / −0.0014 EpochAvg on grasp. `reconstruction_loss_weight` was 4.0,
so this is not a weight-too-small result. But inside the s1_b2048 setting `recon` *does*
help (+0.0149 Last on grasp, +0.0205 Last on slip), consistent with the term mattering
only while a run is under-trained.

**3. `tip` reaches `rope` e100 quality in a fifth of the pretraining, but does not pass
it.** Last has `tip` e50 ahead by 0.005 (inside its 0.018 fold std); EpochAvg has `rope`
ahead by 0.003. Call it a tie at 1/2 the epochs — and note `rope` was trained before the
iBOT fix, so it is not a matched control.

**4. stride 1 + batch 2048 underperforms, but the LR is confounded.** `tip_s1_b2048`
reaches only 0.8453 Last on grasp at epoch 10 — scratch level, 0.0441 below stride-3 `tip`
at the same epoch — despite each epoch covering 3x the windows. The config comment flags
the cause: batch went 512 → 2048 while LR stayed 1e-4, where linear scaling suggests 4e-4.
**No stride conclusion until the scaled-LR rerun exists.**

**5. Slip single-seed rankings are not trustworthy on Last alone.** Fold stds run
0.011–0.050 while the whole slip column spans 0.6177–0.6860. `tip` e30 Last (0.6536) looks
like a collapse next to e20 (0.6860), but its EpochAvg (0.6538) and its best-epoch value
(0.6966, in the logs) show the run was fine and the last epoch simply landed badly.
EpochAvg is the more stable read on this task.

## Caveats

- **Single seed (0) throughout.** Three-seed reference for scratch:
  grasp 0.8475 ± 0.0067, slip 0.6583 ± 0.0096 (`results_ropeprobe_seeds_20260728_111602.csv`).
- `rope` epoch 100 predates the iBOT masked-token indexing fix (iBOT saw ~2% of each
  batch), so it is not a matched control for anything below it.
- One backbone LR and one probe LR; neither swept for the 890k-parameter probe.
- ID protocol only (episode-level K-fold over all objects); no leave-one-object-out.

## Suggested next steps

1. **Three seeds for `tip` epoch 20 and `recon_tip_s1_b2048` epoch 10** — the two leaders,
   one per metric on slip. 8 runs, ~30 min on 4 GPUs.
2. **Rerun `tip_s1_b2048` at LR 4e-4** before drawing any stride conclusion.
3. Matched `ibotfix` epochs 20/30/50 to confirm the fingertip gap holds past epoch 10.
