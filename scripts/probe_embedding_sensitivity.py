"""Probe whether encoder embeddings are driven by fingertip xyz or by force.

For batches of real frames the script recomputes embeddings after perturbing
one input stream at a time and measures the mean cosine distance to the
unperturbed embedding:

  force_shuffle    sensor rows permuted across the batch (xyz fixed)
  xyz_shuffle      xyz rows permuted across the batch (sensor fixed)
  both_shuffle     independent permutations of both streams (upper reference)
  force_zero       sensor set to all-zero (no contact)
  xyz_mean         xyz set to the batch-mean pose
  ch0/ch12/ch3     only that force channel group shuffled
                   (ch0 normal, ch12 tangential xy, ch3 proximity)

A large distance means the embedding depends strongly on that stream.
Frames come from the raw-comparison caches (pretrain / slip / grasp).

Usage:
  XFORMERS_DISABLED=TRUE python scripts/probe_embedding_sensitivity.py \
      --checkpoints <ckpt1> <ckpt2>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from compare_pretrain_downstream_raw import (  # noqa: E402
    iter_pretrain_episodes, iter_downstream_episodes, pick_spread, load_source,
)
from compare_embeddings import build_encoder, preprocess, embed  # noqa: E402

DEFAULT_CKPTS = [
    "experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1_ibotfix/"
    "2026.07.28-12-27/checkpoints/epoch-0010.ckpt",
    "experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1/"
    "2026.07.27-22-14/checkpoints/epoch-0100.ckpt",
]

SOURCE_COLORS = {"pretrain": "#0072B2", "slip": "#E69F00", "grasp": "#009E73"}


def perturbations(sensor, pos, seed=0):
    """Yield (name, sensor_variant, pos_variant) for a fixed frame batch."""
    g = torch.Generator().manual_seed(seed)
    perm1 = torch.randperm(len(sensor), generator=g)
    perm2 = torch.randperm(len(sensor), generator=g)

    yield "force_shuffle", sensor[perm1], pos
    yield "xyz_shuffle", sensor, pos[perm1]
    yield "both_shuffle", sensor[perm1], pos[perm2]
    yield "force_zero", torch.zeros_like(sensor), pos
    yield "xyz_mean", sensor, pos.mean(dim=0, keepdim=True).expand_as(pos).contiguous()
    for name, chs in (("ch0_shuffle", [0]), ("ch12_shuffle", [1, 2]), ("ch3_shuffle", [3])):
        s = sensor.clone()
        s[..., chs] = sensor[perm1][..., chs]
        yield name, s, pos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CKPTS)
    ap.add_argument("--pretrain_root", default="pretraining_dataset/brainco")
    ap.add_argument("--slip_root", default="dataset/brainco/downstream/slip_data_v2")
    ap.add_argument("--grasp_root", default="dataset/brainco/downstream/grasp_prediction")
    ap.add_argument("--urdf", default="dataset/brainco/urdf")
    ap.add_argument("--n_frames", type=int, default=2000, help="frames sampled per source")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--raw_cache_pretrain", default="scripts/compare_raw_out")
    ap.add_argument("--raw_cache_slip", default="scripts/compare_raw_out_slip")
    ap.add_argument("--raw_cache_grasp", default="scripts/compare_raw_out")
    ap.add_argument("--out", default="scripts/compare_emb_sensitivity")
    ap.add_argument("--data_stats", action="store_true",
                    help="Override encoder signal_mean/std with statistics computed "
                         "from each source's own frames (mimics the downstream "
                         "training-split normalization) before measuring sensitivity.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    urdf = Path(args.urdf)

    # same episode selections as the raw comparison -> cache hits
    eps_pt = pick_spread(iter_pretrain_episodes(Path(args.pretrain_root)), 12)
    eps_sl = pick_spread(iter_downstream_episodes(Path(args.slip_root)), 24)
    eps_gr = pick_spread(iter_downstream_episodes(Path(args.grasp_root)), 30)
    sources_raw = {
        "pretrain": load_source("pretrain", eps_pt, urdf, 10, Path(args.raw_cache_pretrain)),
        "slip": load_source("downstream", eps_sl, urdf, 2, Path(args.raw_cache_slip)),
        "grasp": load_source("downstream", eps_gr, urdf, 1, Path(args.raw_cache_grasp)),
    }

    rng = np.random.default_rng(0)
    inputs = {}
    for name, (xyz, tac, _) in sources_raw.items():
        idx = rng.choice(len(xyz), min(args.n_frames, len(xyz)), replace=False)
        sensor, pos = preprocess(tac[idx], xyz[idx])
        inputs[name] = (sensor, pos)

    results = {}  # (ckpt_label, source) -> {pert: mean cos dist}
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        label = f"{ckpt_path.parents[2].name}/{ckpt_path.stem}"
        encoder = build_encoder(ckpt_path, args.device)
        for src, (sensor, pos) in inputs.items():
            if args.data_stats:
                flat = sensor.reshape(-1, sensor.shape[-1])
                mean = flat.mean(dim=0)
                std = flat.std(dim=0).clamp(min=1e-6)
                encoder.update_stats(mean.to(args.device), std.to(args.device))
                print(f"[{label}] {src}: data stats mean={mean.tolist()} std={std.tolist()}")
            base = F.normalize(embed(encoder, sensor, pos, args.batch_size, args.device), dim=-1)
            row = {}
            for pert_name, s_v, p_v in perturbations(sensor, pos):
                e = F.normalize(embed(encoder, s_v, p_v, args.batch_size, args.device), dim=-1)
                row[pert_name] = float((1.0 - (base * e).sum(dim=-1)).mean())
            results[(label, src)] = row
            share = row["force_shuffle"] / (row["force_shuffle"] + row["xyz_shuffle"] + 1e-12)
            print(f"[{label}] {src}: force={row['force_shuffle']:.4f} "
                  f"xyz={row['xyz_shuffle']:.4f} (force share {share:.2f})")
        del encoder
        torch.cuda.empty_cache()

    pert_names = list(next(iter(results.values())).keys())

    # ── summary ──
    md = [
        "# Embedding sensitivity: force vs fingertip xyz",
        "",
        f"- metric: mean cosine distance between embeddings of real frames and the same",
        "  frames with ONE input stream perturbed (shuffled across the batch / zeroed).",
        f"- {args.n_frames} frames per source; frame sources as in the raw comparison.",
        "",
    ]
    for label in dict.fromkeys(l for l, _ in results):
        md += [f"## {label}", "",
               "| source | " + " | ".join(pert_names) + " | force share* |",
               "|---|" + "---|" * (len(pert_names) + 1)]
        for src in inputs:
            row = results[(label, src)]
            share = row["force_shuffle"] / (row["force_shuffle"] + row["xyz_shuffle"] + 1e-12)
            md.append(f"| {src} | " + " | ".join(f"{row[p]:.4f}" for p in pert_names)
                      + f" | {share:.2f} |")
        md.append("")
    md += ["*force share = force_shuffle / (force_shuffle + xyz_shuffle); "
           "0.5 = equally sensitive, 1.0 = force-only.", ""]
    (out_dir / "summary.md").write_text("\n".join(md))
    print(f"wrote {out_dir / 'summary.md'}")

    # ── figure ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_labels = list(dict.fromkeys(l for l, _ in results))
    fig, axes = plt.subplots(1, len(ckpt_labels), figsize=(8.5 * len(ckpt_labels), 4.8),
                             sharey=True, squeeze=False)
    x = np.arange(len(pert_names))
    width = 0.25
    for ax, label in zip(axes[0], ckpt_labels):
        for si, src in enumerate(inputs):
            vals = [results[(label, src)][p] for p in pert_names]
            ax.bar(x + (si - 1) * width, vals, width * 0.92,
                   color=SOURCE_COLORS[src], label=src)
        ax.set_xticks(x)
        ax.set_xticklabels(pert_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].set_ylabel("mean cosine distance to unperturbed embedding")
    axes[0][0].legend(fontsize=9)
    fig.suptitle("Embedding sensitivity to perturbing one input stream")
    fig.savefig(out_dir / "fig_sensitivity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'fig_sensitivity.png'}")


if __name__ == "__main__":
    main()
