#!/usr/bin/env python
"""Re-run a trained slip-detection run and dump a sample of misclassified windows.

Rebuilds the same episode-level K-fold split, restores each fold's saved weights,
runs validation inference, collects every misclassified window, then writes a
random sample of them to an output folder as one PNG per window plus a manifest
CSV and a summary.

Usage:
    python scripts/dump_slip_misclassified.py \
        --run-dir experiments/brainco_xyz_slip_detection/2026.07.28_16-34_nobal_e30_tip_s3 \
        --out scripts/misclassified/tip_s3 --n 30
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CH = ["Fn", "Ft*cos", "Ft*sin", "prox"]


def build_split(dataset, fold, num_folds, split_seed):
    """Same episode-level split as train_task_brainco_angle.get_dataloader_*."""
    ep_start, cur = {}, 0
    for ep in dataset.episode_data:
        ep_start[ep["path"]] = cur
        cur += len(ep["window_starts"])
    episodes = list(dataset.episode_data)
    random.Random(split_seed).shuffle(episodes)
    n = len(episodes)
    size = n // num_folds
    lo = fold * size
    hi = lo + size if fold < num_folds - 1 else n
    train_idx, val_idx = [], []
    for rank, ep in enumerate(episodes):
        wins = list(range(ep_start[ep["path"]], ep_start[ep["path"]] + len(ep["window_starts"])))
        (val_idx if lo <= rank < hi else train_idx).extend(wins)
    return train_idx, val_idx


def channel_stats(subset, key="joint_contact"):
    """Population channel mean/std over a training subset (matches training)."""
    s = sq = None
    n = 0
    for i in range(len(subset)):
        v = torch.as_tensor(subset[i][key], dtype=torch.float64).reshape(-1, 4)
        if s is None:
            s = torch.zeros(4, dtype=torch.float64)
            sq = torch.zeros(4, dtype=torch.float64)
        s += v.sum(0)
        sq += v.square().sum(0)
        n += v.shape[0]
    mean = s / n
    std = (sq / n - mean.square()).clamp_min(0).sqrt()
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    return mean.float(), std.float()


def load_rgb(episode_path, frame_idx):
    """Colors are 1:1 with the interpolated frame index (verified 407<->407)."""
    import cv2
    d = Path(episode_path) / "colors"
    if not d.exists():
        return None
    files = sorted(f for f in d.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        return None
    bgr = cv2.imread(str(files[min(frame_idx, len(files) - 1)]))
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


MAX_RGB = 6


def plot_window(sample, pred, out_path, title, episode_path, start_frame):
    contact = sample["joint_contact"].numpy()      # (W, 10, 4)
    xyz = sample["finger_xyz"].numpy()             # (W, 10, 3)
    w = contact.shape[0]
    # Long windows would need one RGB panel per frame; show an evenly spaced subset.
    rgb_idx = list(range(w)) if w <= MAX_RGB else \
        sorted(set(np.linspace(0, w - 1, MAX_RGB).round().astype(int).tolist()))
    ncol = max(5, len(rgb_idx))
    xticks = range(w) if w <= 8 else range(0, w, max(1, w // 8))

    fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol, 7.2))

    for c in range(4):                             # top row: tactile channels
        ax = axes[0, c]
        im = ax.imshow(contact[:, :, c].T, aspect="auto", cmap="viridis")
        ax.set_title(CH[c]); ax.set_xlabel("frame")
        ax.set_ylabel("sensor" if c == 0 else "")
        ax.set_xticks(xticks)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[0, 4]
    for s in range(xyz.shape[1]):
        ax.plot(range(w), np.linalg.norm(xyz[:, s, :] - xyz[0, s, :], axis=-1), marker="o", ms=3)
    ax.set_title("fingertip displacement (vs frame 0)")
    ax.set_xlabel("frame"); ax.set_xticks(xticks)
    for k in range(5, ncol):
        axes[0, k].axis("off")

    for col in range(ncol):                        # bottom row: RGB frames
        ax = axes[1, col]
        ax.axis("off")
        if col >= len(rgb_idx):
            continue
        i = rgb_idx[col]
        rgb = load_rgb(episode_path, start_frame + i)
        if rgb is not None:
            ax.imshow(rgb)
        else:
            ax.text(0.5, 0.5, "no RGB", ha="center", va="center")
        ax.set_title(f"frame {start_frame + i}", fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30, help="misclassified samples")
    ap.add_argument("--n-correct", type=int, default=30, help="correctly classified samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    num_folds = int(cfg.get("num_folds", 4))
    split_seed = int(cfg.get("split_seed", 42))

    OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
    OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)
    OmegaConf.register_new_resolver("capitalize", lambda s: s.title(), replace=True)

    dataset = hydra.utils.instantiate(cfg.data.dataset)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for fold in range(num_folds):
        ck = run_dir / f"checkpoints_fold{fold}" / "last.ckpt"
        if not ck.exists():
            print(f"fold {fold}: no checkpoint, skipping")
            continue
        train_idx, val_idx = build_split(dataset, fold, num_folds, split_seed)
        mean, std = channel_stats(data.Subset(dataset, train_idx))

        model = hydra.utils.instantiate(cfg.task)
        model.model_encoder.update_stats(mean, std)
        state = torch.load(ck, map_location="cpu", weights_only=False)
        sd = state.get("model", state.get("state_dict", state))
        missing = model.load_state_dict(sd, strict=False)
        model.to(dev).eval()

        loader = data.DataLoader(data.Subset(dataset, val_idx), batch_size=256, shuffle=False)
        seen = 0
        with torch.no_grad():
            for batch in loader:
                gpu_batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                logits = model.forward(gpu_batch)
                pred = logits.argmax(-1).cpu()
                label = batch["label"]
                for i in range(len(pred)):
                    rows.append({
                        "fold": fold,
                        "global_index": val_idx[seen + i],
                        "episode": batch["episode_path"][i],
                        "window_start_frame": int(batch["window_start_frame"][i]),
                        "true": int(label[i]),
                        "pred": int(pred[i]),
                        "correct": int(pred[i] == label[i]),
                    })
                seen += len(pred)
        n_wrong = sum(r["fold"] == fold and not r["correct"] for r in rows)
        print(f"fold {fold}: {len(val_idx)} val windows, {n_wrong} misclassified "
              f"(missing keys: {len(missing.missing_keys)})")

    out = (PROJECT_ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    KIND = {(1, 0): "FN_slip_missed", (0, 1): "FP_false_alarm",
            (1, 1): "TP_slip_hit", (0, 0): "TN_nonslip_hit"}
    picked_all = []
    for bucket, want in (("misclassified", args.n), ("correct", args.n_correct)):
        pool = [r for r in rows if r["correct"] == (bucket == "correct")]
        if not pool or want <= 0:
            continue
        sub = out / bucket
        sub.mkdir(exist_ok=True)
        picked = rng.sample(pool, min(want, len(pool)))
        picked.sort(key=lambda r: (r["fold"], r["episode"], r["window_start_frame"]))
        for rank, r in enumerate(picked):
            sample = dataset[r["global_index"]]
            kind = KIND[(r["true"], r["pred"])]
            ep = Path(r["episode"])
            name = f"{rank:02d}_{kind}_{ep.parent.name}_{ep.name}_f{r['window_start_frame']}.png"
            plot_window(sample, r["pred"], sub / name,
                        f"{kind}  |  {ep.parent.name}/{ep.name}  frame {r['window_start_frame']}  "
                        f"|  true={r['true']} pred={r['pred']}  (fold {r['fold']})",
                        r["episode"], r["window_start_frame"])
            r["png"] = f"{bucket}/{name}"
            r["bucket"] = bucket
        picked_all += picked
        print(f"{bucket}: wrote {len(picked)} PNGs to {sub}")

    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bucket", "png", "fold", "episode",
                                           "window_start_frame", "true", "pred",
                                           "correct", "global_index"])
        w.writeheader()
        w.writerows(picked_all)

    wrong = [r for r in rows if not r["correct"]]
    fn = sum(r["true"] == 1 for r in wrong)
    fp = len(wrong) - fn
    tp = sum(r["true"] == 1 and r["correct"] for r in rows)
    tn = sum(r["true"] == 0 and r["correct"] for r in rows)
    summary = [
        f"run: {run_dir.name}",
        f"val windows: {len(rows)}   correct: {len(rows)-len(wrong)}   wrong: {len(wrong)}",
        f"  TP (slip hit)      {tp}",
        f"  TN (non-slip hit)  {tn}",
        f"  FN (slip missed)   {fn} ({100*fn/max(len(wrong),1):.1f}% of errors)",
        f"  FP (false alarm)   {fp} ({100*fp/max(len(wrong),1):.1f}% of errors)",
        f"sampled (seed {args.seed}): "
        f"{sum(r['bucket']=='misclassified' for r in picked_all)} wrong, "
        f"{sum(r['bucket']=='correct' for r in picked_all)} correct",
    ]
    (out / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
