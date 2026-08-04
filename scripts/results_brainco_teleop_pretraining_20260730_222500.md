# HOI x BrainCo pretraining ablation

Generated 2026-07-30 23:55

## Protocol

| | |
| --- | --- |
| Entry point | `train_task_brainco_angle.py` |
| grasp | `task/grasp_prediction/temporal_w15_cls_d4_fe4`, 4-fold CV |
| slip | `task/slip_detection/temporal_w15_cls_d4_fe4_v3` (`slip_data_v3`), 3-fold CV, stratified by episode index |
| Overrides | `task.encoder_lr=1e-4`, `++split_seed=42`, 100 downstream epochs, batch 256 |
| Seeds | 0, 1 — `±` is the spread across the two |
| Metric | balanced accuracy |
| Exception | `HOI jointonly` uses the `*_jointonly` task configs (`input_streams: pos`); its checkpoint has no trained sensor path |

Every row was measured under this one protocol. Numbers for the same encoders in
older `results_seed_ood_*.md` files differ (readout, fold count, slip data
version and seed set all vary there) and are not mixed in.

## Models

| # | HOI | BrainCo | Model | Pretraining input | in_dim | HOI lr/ep | BrainCo lr/warmup/ep |
| ---: | :---: | :---: | --- | --- | ---: | --- | --- |
| 1 | ✗ | ✗ | no pretraining | — | — | — | — |
| 2 | ✓ | ✗ | HOI tip | proximity | 10 | 4e-4 / 500 | — |
| 3 | ✓ | ✗ | HOI 42j | proximity | 42 | 4e-4 / 500 | — |
| 4 | ✓ | ✗ | HOI jointonly | none (force removed) | 42 | 4e-4 / 500 | — |
| 5 | ✗ | ✓ | brainco-only SSL | Fn / Ft·cos / Ft·sin / prox | 10 | — | 4e-4 / 10 / 1000 |
| 6 | ✓ | ✓ | hoi-init SSL | Fn / Ft·cos / Ft·sin / prox | 10 | 4e-4 / 500 | 4e-4 / 10 / 1000 |
| 7 | ✓ | ✓ | hoi-init gentle 5e-5 | Fn / Ft·cos / Ft·sin / prox | 10 | 4e-4 / 500 | 5e-5 / **0** / 100 |
| 8 | ✓ | ✓ | hoi-init gentle 1e-5 | Fn / Ft·cos / Ft·sin / prox | 10 | 4e-4 / 500 | 1e-5 / **0** / 100 |

All BrainCo runs start from `temporal_d4fe4cls_tip_fdino_lr4e4_ep0500` (= model 2)
when HOI is ✓, with `sensor_embed` left at random init.

## Balanced accuracy — Best Epoch

| # | HOI | BrainCo | Model | grasp | slip |
| ---: | :---: | :---: | --- | ---: | ---: |
| 1 | ✗ | ✗ | no pretraining | 0.8615 ± 0.0022 | 0.7519 ± 0.0036 |
| 2 | ✓ | ✗ | HOI tip | 0.9411 ± 0.0072 | 0.7649 ± 0.0008 |
| 3 | ✓ | ✗ | HOI 42j | 0.9370 ± 0.0021 | 0.7539 ± 0.0078 |
| 4 | ✓ | ✗ | HOI jointonly | 0.8914 ± 0.0016 | 0.5446 ± 0.0204 |
| 5 | ✗ | ✓ | brainco-only SSL | 0.8790 ± 0.0003 | 0.7368 ± 0.0035 |
| 6 | ✓ | ✓ | hoi-init SSL | 0.9008 ± 0.0071 | 0.7697 ± 0.0079 |
| 7 | ✓ | ✓ | hoi-init gentle 5e-5 | 0.9405 ± 0.0025 | 0.7629 ± 0.0103 |
| 8 | ✓ | ✓ | **hoi-init gentle 1e-5** | **0.9462 ± 0.0026** | **0.7704 ± 0.0062** |

## Balanced accuracy — Last Epoch

| # | Model | grasp | slip |
| ---: | --- | ---: | ---: |
| 1 | no pretraining | 0.8467 ± 0.0015 | 0.7097 ± 0.0023 |
| 2 | HOI tip | 0.9363 ± 0.0074 | 0.7121 ± 0.0040 |
| 3 | HOI 42j | 0.9270 ± 0.0017 | 0.7243 ± 0.0108 |
| 4 | HOI jointonly | 0.8749 ± 0.0025 | 0.5280 ± 0.0000 |
| 5 | brainco-only SSL | 0.8712 ± 0.0047 | 0.7228 ± 0.0019 |
| 6 | hoi-init SSL | 0.8882 ± 0.0113 | **0.7382 ± 0.0054** |
| 7 | hoi-init gentle 5e-5 | 0.9328 ± 0.0073 | 0.7305 ± 0.0029 |
| 8 | hoi-init gentle 1e-5 | **0.9381 ± 0.0016** | 0.7237 ± 0.0074 |

## Balanced accuracy — Epoch Average

| # | Model | grasp | slip |
| ---: | --- | ---: | ---: |
| 1 | no pretraining | 0.8409 ± 0.0003 | 0.7072 ± 0.0052 |
| 2 | HOI tip | 0.9300 ± 0.0062 | 0.7177 ± 0.0007 |
| 3 | HOI 42j | 0.9222 ± 0.0022 | 0.7282 ± 0.0085 |
| 4 | HOI jointonly | 0.8681 ± 0.0014 | 0.5261 ± 0.0007 |
| 5 | brainco-only SSL | 0.8673 ± 0.0049 | 0.7210 ± 0.0009 |
| 6 | hoi-init SSL | 0.8822 ± 0.0085 | **0.7364 ± 0.0023** |
| 7 | hoi-init gentle 5e-5 | 0.9269 ± 0.0059 | 0.7318 ± 0.0021 |
| 8 | hoi-init gentle 1e-5 | **0.9331 ± 0.0006** | 0.7300 ± 0.0025 |

## Macro F1 — Best Epoch

| # | Model | grasp | slip |
| ---: | --- | ---: | ---: |
| 1 | no pretraining | 0.8595 | 0.7346 |
| 2 | HOI tip | 0.9409 | 0.7514 |
| 3 | HOI 42j | 0.9371 | 0.7499 |
| 4 | HOI jointonly | 0.8916 | 0.5105 |
| 5 | brainco-only SSL | 0.8767 | 0.7337 |
| 6 | hoi-init SSL | 0.9005 | 0.7582 |
| 7 | hoi-init gentle 5e-5 | 0.9402 | 0.7510 |
| 8 | hoi-init gentle 1e-5 | **0.9448** | **0.7523** |

## BrainCo SSL strength sweep, from the same HOI tip checkpoint

Best Epoch balanced accuracy. lr 0 is model 2 (no BrainCo SSL at all).

| BrainCo lr | warmup | epochs | grasp | slip |
| ---: | :---: | ---: | ---: | ---: |
| 4e-4 | 10 | 1000 | 0.9008 | 0.7697 |
| 4e-4 | 10 | 100 | 0.8958 <sub>1 seed</sub> | 0.7776 <sub>1 seed</sub> |
| 5e-5 | 0 | 100 | 0.9405 | 0.7629 |
| 1e-5 | 0 | 100 | **0.9462** | **0.7704** |
| 0 | — | 0 | 0.9411 | 0.7649 |

## Findings

| | |
| --- | --- |
| The grasp erosion was an lr artefact | 4e-4 costs 4.0 points against the untouched HOI checkpoint (0.9008 vs 0.9411); 1e-5 recovers and marginally exceeds it (0.9462). Monotone in lr. |
| Gentle BrainCo SSL is the best arm on both tasks | Model 8 leads grasp (+0.005 over HOI tip) and slip (+0.006). Both margins are inside or at the edge of the seed spread, so read it as "does not hurt, may help slightly", not as a clear win. |
| BrainCo SSL alone beats nothing on grasp, loses on slip | Model 5 vs 1: grasp +1.8 points, slip **−1.5**. 44 min of single-embodiment teleop is not a substitute for HOI (−6.2 on grasp). |
| Force matters for slip, much less for grasp | Model 4 (no force at all) holds 0.8914 on grasp — 2.9 above no-pretraining — but collapses to 0.5446 on slip, near chance for the 6-class problem. |
| tip vs 42-joint is a wash | Models 2 and 3 differ by 0.004 on grasp and 0.011 on slip, comparable to the seed spread. |
| Readout choice changes the slip ranking | Model 8 leads slip at Best Epoch but model 6 leads at Last Epoch and Epoch Average. grasp is stable across all three readouts. |

## Caveats

- 2 seeds. Fold-to-fold spread (0.009–0.029) is several times the seed spread,
  so fold assignment is the larger noise source; treat differences under ~0.01
  as unresolved.
- Models 2–4 load 126/129 encoder tensors — `signal_mean`, `signal_std` and
  `sensor_embed.proj.weight` are dropped (proximity-only checkpoints, in_chans
  1 → 4) and the force projection is relearned downstream. Models 5–8 load all
  129. This is the standard treatment for prox checkpoints in this repo, but it
  does mean the HOI-only arms and the BrainCo arms differ in one more way than
  the ✓/✗ grid suggests.
- Models 7 and 8 change lr, warmup and epoch budget together against model 6,
  so the sweep table attributes to the combination.
- One downstream backbone lr (1e-4), ID protocol only, no OOD.
- BrainCo pretraining used all 12 episodes with no held-out split, so its
  pretraining loss carries no generalisation signal.
- The pretraining data has a stuck right-hand sensor 3 in 5 of 12 episodes and
  near-dead fingertips in `driver`; `subtract_baseline` is not applied.
