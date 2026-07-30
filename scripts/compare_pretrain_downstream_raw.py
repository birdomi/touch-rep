"""Compare raw BrainCo pretraining vs downstream data.

For both sources this script loads, per frame and per fingertip sensor:
  - the raw 4-channel tactile value  [ch0=Fn, ch1=Ft, ch2=direction(65535->-1), ch3=proximity]
  - the fingertip xyz position (wrist/base_link-relative, via compute_fk)

It then matches each downstream (sensor, frame) point to its nearest
pretraining neighbours in fingertip-xyz space (same sensor index only)
and compares the tactile values of the matched pairs.

Outputs (in --out):
  summary.md                    stats tables (per-channel, NN distances, matched diffs)
  fig_channel_hists.png         raw channel distributions, pretrain vs downstream
  fig_xyz_planes.png            fingertip xyz coverage (XY/XZ/YZ), both sources
  fig_xyz_per_sensor.png        per-sensor XY coverage
  fig_nn_dist.png               per-sensor NN distance (downstream -> pretrain)
  fig_matched_tactile.png       matched-pair tactile scatter (downstream vs NN pretrain)
  cache_*.npz                   loaded raw arrays (reused on re-run)

Usage:
  python scripts/compare_pretrain_downstream_raw.py \
      --pretrain_root pretraining_dataset/brainco \
      --downstream_root dataset/brainco/downstream/grasp_prediction \
      --max_episodes_pretrain 12 --max_episodes_downstream 30
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tactile_ssl.data.brainco_tactile import compute_fk  # noqa: E402

SENSOR_NAMES = [
    "L_thumb", "L_index", "L_middle", "L_ring", "L_pinky",
    "R_thumb", "R_index", "R_middle", "R_ring", "R_pinky",
]
CHANNEL_NAMES = ["ch0 normal_force", "ch1 tangential_force", "ch2 direction", "ch3 proximity"]

# Okabe-Ito (CVD-safe): blue = pretrain, orange = downstream
C_PT = "#0072B2"
C_DS = "#E69F00"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_episode_raw(ep_path: Path, urdf_path: Path, stride: int):
    """Return (xyz (F,10,3), tac (F,10,4)) for one episode, frames subsampled by stride."""
    with open(ep_path / "data.json") as f:
        frames = json.load(f)["data"][::stride]

    tac_list = []
    for fr in frames:
        halves = []
        for side in ("left_ee", "right_ee"):
            t = fr["tactiles"][side]
            if isinstance(t, str):
                arr = np.load(str(ep_path / t)).reshape(-1, 4)
            else:
                arr = np.array(t).reshape(-1, 4)
            halves.append(arr)
        tac_list.append(np.concatenate(halves, axis=0))
    tac = np.array(tac_list, dtype=np.float32)          # (F, 10, 4)
    tac[..., 2][tac[..., 2] == 65535] = -1              # invalid direction -> -1

    fk = compute_fk(ep_path, urdf_path, frames)
    xyz = fk["fingertip_base"].astype(np.float32)       # (F, 10, 3) wrist-relative
    return xyz, tac


def iter_pretrain_episodes(root: Path):
    for obj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ep in sorted(p for p in obj_dir.iterdir() if p.is_dir() and (p / "data.json").exists()):
            yield f"{obj_dir.name}/{ep.name}", ep


def iter_downstream_episodes(root: Path):
    """Yield episodes under class subdirs (grasp_success/...) or directly under root."""
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    direct = [p for p in subdirs if (p / "data.json").exists()]
    if direct:
        for ep in direct:
            yield ep.name, ep
        return
    for cls_dir in subdirs:
        for ep in sorted(p for p in cls_dir.iterdir() if p.is_dir() and (p / "data.json").exists()):
            yield f"{cls_dir.name}/{ep.name}", ep


def pick_spread(items, k):
    """Pick up to k items evenly spread over the list (keeps object/class diversity)."""
    items = list(items)
    if len(items) <= k:
        return items
    idxs = np.unique(np.linspace(0, len(items) - 1, k).round().astype(int))
    return [items[i] for i in idxs]


def load_source(name, episodes, urdf_path, stride, cache_dir):
    key = hashlib.md5(
        (";".join(n for n, _ in episodes) + f"|{stride}").encode()
    ).hexdigest()[:10]
    cache = cache_dir / f"cache_{name}_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"[{name}] cache hit: {cache.name}  ({z['xyz'].shape[0]} frames)")
        return z["xyz"], z["tac"], list(z["ep_names"])

    xyz_all, tac_all, ep_names = [], [], []
    for i, (ep_name, ep_path) in enumerate(episodes):
        print(f"[{name}] ({i + 1}/{len(episodes)}) loading {ep_name} ...")
        xyz, tac = load_episode_raw(ep_path, urdf_path, stride)
        xyz_all.append(xyz)
        tac_all.append(tac)
        ep_names.append(ep_name)
    xyz = np.concatenate(xyz_all, axis=0)
    tac = np.concatenate(tac_all, axis=0)
    np.savez_compressed(cache, xyz=xyz, tac=tac, ep_names=np.array(ep_names, dtype=object))
    print(f"[{name}] loaded {xyz.shape[0]} frames from {len(ep_names)} episodes -> {cache.name}")
    return xyz, tac, ep_names


# ── Analysis ─────────────────────────────────────────────────────────────────

def channel_stats(tac):
    """tac (F,10,4) -> dict[ch] of stats over all frames*sensors (ch2: valid only)."""
    rows = {}
    flat = tac.reshape(-1, 4)
    for c in range(4):
        v = flat[:, c]
        if c == 2:
            valid = v[v >= 0]
            rows[c] = dict(
                valid_ratio=float((v >= 0).mean()),
                mean=float(valid.mean()) if valid.size else float("nan"),
                std=float(valid.std()) if valid.size else float("nan"),
                p50=float(np.median(valid)) if valid.size else float("nan"),
                p95=float(np.percentile(valid, 95)) if valid.size else float("nan"),
                max=float(valid.max()) if valid.size else float("nan"),
            )
        else:
            rows[c] = dict(
                valid_ratio=1.0,
                mean=float(v.mean()), std=float(v.std()),
                p50=float(np.median(v)), p95=float(np.percentile(v, 95)),
                max=float(v.max()),
            )
    return rows


def match_nearest(xyz_pt, tac_pt, xyz_ds, tac_ds, knn):
    """Per-sensor KDTree match: downstream point -> nearest pretraining points.

    Returns dict of flat arrays over (downstream frame x sensor):
      sensor, nn_dist, ds_tac (M,4), pt_tac (M,4) [mean of knn neighbours' tactile]
    """
    from scipy.spatial import cKDTree

    out = dict(sensor=[], nn_dist=[], ds_tac=[], pt_tac=[])
    for s in range(10):
        tree = cKDTree(xyz_pt[:, s, :])
        dist, idx = tree.query(xyz_ds[:, s, :], k=knn)
        if knn == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        neigh_tac = tac_pt[idx, s, :]                    # (M, knn, 4)
        # ch2: average valid entries only
        pt_mean = neigh_tac.mean(axis=1)
        ch2 = neigh_tac[..., 2]
        ch2_valid = ch2 >= 0
        with np.errstate(invalid="ignore"):
            ch2_mean = np.where(
                ch2_valid.any(axis=1),
                np.where(ch2_valid, ch2, 0).sum(axis=1) / np.maximum(ch2_valid.sum(axis=1), 1),
                -1.0,
            )
        pt_mean[:, 2] = ch2_mean
        out["sensor"].append(np.full(len(xyz_ds), s))
        out["nn_dist"].append(dist[:, 0])
        out["ds_tac"].append(tac_ds[:, s, :])
        out["pt_tac"].append(pt_mean)
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# ── Report ───────────────────────────────────────────────────────────────────

def fmt_stats_table(stats_pt, stats_ds):
    lines = [
        "| channel | source | valid% | mean | std | p50 | p95 | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in range(4):
        for src, st in (("pretrain", stats_pt[c]), ("downstream", stats_ds[c])):
            lines.append(
                f"| {CHANNEL_NAMES[c]} | {src} | {st['valid_ratio'] * 100:.1f} "
                f"| {st['mean']:.1f} | {st['std']:.1f} | {st['p50']:.1f} "
                f"| {st['p95']:.1f} | {st['max']:.1f} |"
            )
    return "\n".join(lines)


def write_summary(out_dir, args, n_pt, n_ds, eps_pt, eps_ds,
                  stats_pt, stats_ds, match, close_mask, thr):
    md = [
        "# Raw data comparison: pretraining vs downstream",
        "",
        f"- pretrain: `{args.pretrain_root}` — {len(eps_pt)} episodes, {n_pt} frames "
        f"(stride {args.frame_stride_pretrain})",
        f"- downstream: `{args.downstream_root}` — {len(eps_ds)} episodes, {n_ds} frames "
        f"(stride {args.frame_stride_downstream})",
        f"- fingertip xyz: wrist(base_link)-relative, via `compute_fk`",
        f"- NN matching: per-sensor KDTree on xyz, k={args.knn}; "
        f"'close pair' threshold = {thr * 1000:.1f} mm",
        "",
        "## Raw tactile channel stats (all frames x sensors)",
        "",
        fmt_stats_table(stats_pt, stats_ds),
        "",
        "## Fingertip-xyz NN distance (downstream -> nearest pretrain, same sensor)",
        "",
        "| sensor | p50 (mm) | p95 (mm) | max (mm) | close pairs (<thr) |",
        "|---|---|---|---|---|",
    ]
    for s in range(10):
        m = match["sensor"] == s
        d = match["nn_dist"][m] * 1000
        md.append(
            f"| {SENSOR_NAMES[s]} | {np.median(d):.2f} | {np.percentile(d, 95):.2f} "
            f"| {d.max():.2f} | {(close_mask & m).sum()}/{m.sum()} |"
        )

    md += [
        "",
        f"## Matched-pair tactile diff (close pairs only, {close_mask.sum()} pairs)",
        "",
        "downstream value vs mean of its k nearest pretraining neighbours:",
        "",
        "| channel | mean abs diff | p50 abs diff | p95 abs diff | corr |",
        "|---|---|---|---|---|",
    ]
    ds_t = match["ds_tac"][close_mask]
    pt_t = match["pt_tac"][close_mask]
    for c in range(4):
        a, b = ds_t[:, c], pt_t[:, c]
        if c == 2:
            v = (a >= 0) & (b >= 0)
            a, b = a[v], b[v]
        diff = np.abs(a - b)
        corr = np.corrcoef(a, b)[0, 1] if len(a) > 1 else float("nan")
        md.append(
            f"| {CHANNEL_NAMES[c]} | {diff.mean():.1f} | {np.median(diff):.1f} "
            f"| {np.percentile(diff, 95):.1f} | {corr:.3f} |"
        )

    path = out_dir / "summary.md"
    path.write_text("\n".join(md) + "\n")
    print(f"wrote {path}")


# ── Figures ──────────────────────────────────────────────────────────────────

def make_figures(out_dir, xyz_pt, tac_pt, xyz_ds, tac_ds, match, close_mask, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    def sub(arr, n):
        if len(arr) <= n:
            return arr
        return arr[rng.choice(len(arr), n, replace=False)]

    # 1. channel histograms
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    flat_pt, flat_ds = tac_pt.reshape(-1, 4), tac_ds.reshape(-1, 4)
    for c, ax in enumerate(axes):
        a, b = flat_pt[:, c], flat_ds[:, c]
        if c == 2:
            a, b = a[a >= 0], b[b >= 0]
        lo = min(a.min(), b.min()) if len(a) and len(b) else 0
        hi = max(np.percentile(a, 99.5), np.percentile(b, 99.5)) if len(a) and len(b) else 1
        bins = np.linspace(lo, hi, 60)
        ax.hist(a, bins=bins, density=True, histtype="step", lw=2, color=C_PT, label="pretrain")
        ax.hist(b, bins=bins, density=True, histtype="step", lw=2, color=C_DS, label="downstream")
        ax.set_yscale("log")
        title = CHANNEL_NAMES[c] + (" (valid only)" if c == 2 else "")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        if c == 0:
            ax.legend(fontsize=9)
    fig.suptitle("Raw tactile channel distributions (density, log scale, clipped at p99.5)", y=1.02)
    fig.savefig(out_dir / "fig_channel_hists.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2. xyz planes (all sensors pooled)
    planes = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
    p_pt = sub(xyz_pt.reshape(-1, 3), 20000)
    p_ds = sub(xyz_ds.reshape(-1, 3), 20000)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (i, j, xl, yl) in zip(axes, planes):
        ax.scatter(p_pt[:, i], p_pt[:, j], s=3, c=C_PT, alpha=0.15, lw=0, label="pretrain")
        ax.scatter(p_ds[:, i], p_ds[:, j], s=3, c=C_DS, alpha=0.15, lw=0, label="downstream")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
    leg = axes[0].legend(fontsize=9, markerscale=4)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.suptitle("Fingertip xyz coverage, wrist-relative (all 10 sensors pooled)")
    fig.savefig(out_dir / "fig_xyz_planes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 3. per-sensor XY coverage
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.4), sharex=True, sharey=True)
    for s in range(10):
        ax = axes[s // 5, s % 5]
        a = sub(xyz_pt[:, s, :], 4000)
        b = sub(xyz_ds[:, s, :], 4000)
        ax.scatter(a[:, 0], a[:, 1], s=3, c=C_PT, alpha=0.2, lw=0)
        ax.scatter(b[:, 0], b[:, 1], s=3, c=C_DS, alpha=0.2, lw=0)
        ax.set_title(SENSOR_NAMES[s], fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Per-sensor fingertip XY coverage — blue: pretrain, orange: downstream")
    fig.savefig(out_dir / "fig_xyz_per_sensor.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 4. NN distance per sensor
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.4), sharex=True)
    for s in range(10):
        ax = axes[s // 5, s % 5]
        d = match["nn_dist"][match["sensor"] == s] * 1000
        ax.hist(d, bins=50, color=C_PT, alpha=0.85)
        ax.axvline(thr * 1000, color="#D55E00", lw=1.5, ls="--")
        ax.set_title(f"{SENSOR_NAMES[s]}  p50={np.median(d):.1f}mm", fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle("NN distance: downstream fingertip -> nearest pretraining fingertip "
                 "(dashed: close-pair threshold)")
    fig.savefig(out_dir / "fig_nn_dist.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 5. matched-pair tactile scatter (close pairs)
    ds_t = match["ds_tac"][close_mask]
    pt_t = match["pt_tac"][close_mask]
    show = [0, 1, 3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, c in zip(axes, show):
        a = sub(np.stack([ds_t[:, c], pt_t[:, c]], axis=1), 20000)
        hi = max(np.percentile(a[:, 0], 99.5), np.percentile(a[:, 1], 99.5), 1)
        ax.scatter(a[:, 0], a[:, 1], s=4, c=C_PT, alpha=0.2, lw=0)
        ax.plot([0, hi], [0, hi], c="#D55E00", lw=1.2, ls="--")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("downstream value")
        ax.set_ylabel("pretrain NN value (k-mean)")
        ax.set_title(CHANNEL_NAMES[c], fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Matched-pair tactile values at similar fingertip xyz "
                 f"(NN dist < {thr * 1000:.0f} mm, {close_mask.sum()} pairs; dashed: y=x)")
    fig.savefig(out_dir / "fig_matched_tactile.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote figures to {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrain_root", default="pretraining_dataset/brainco")
    ap.add_argument("--downstream_root", default="dataset/brainco/downstream/grasp_prediction")
    ap.add_argument("--urdf", default="dataset/brainco/urdf")
    ap.add_argument("--max_episodes_pretrain", type=int, default=12)
    ap.add_argument("--max_episodes_downstream", type=int, default=30)
    ap.add_argument("--frame_stride_pretrain", type=int, default=10)
    ap.add_argument("--frame_stride_downstream", type=int, default=1)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--close_thr_mm", type=float, default=5.0,
                    help="NN distance threshold (mm) for 'close pair' tactile comparison")
    ap.add_argument("--out", default="scripts/compare_raw_out")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    urdf = Path(args.urdf)
    thr = args.close_thr_mm / 1000.0

    eps_pt = pick_spread(iter_pretrain_episodes(Path(args.pretrain_root)), args.max_episodes_pretrain)
    eps_ds = pick_spread(iter_downstream_episodes(Path(args.downstream_root)), args.max_episodes_downstream)
    print(f"pretrain episodes ({len(eps_pt)}): {[n for n, _ in eps_pt]}")
    print(f"downstream episodes ({len(eps_ds)}): {[n for n, _ in eps_ds]}")

    xyz_pt, tac_pt, names_pt = load_source("pretrain", eps_pt, urdf, args.frame_stride_pretrain, out_dir)
    xyz_ds, tac_ds, names_ds = load_source("downstream", eps_ds, urdf, args.frame_stride_downstream, out_dir)

    stats_pt = channel_stats(tac_pt)
    stats_ds = channel_stats(tac_ds)

    print("matching downstream -> pretrain (per-sensor KDTree on fingertip xyz) ...")
    match = match_nearest(xyz_pt, tac_pt, xyz_ds, tac_ds, args.knn)
    close_mask = match["nn_dist"] < thr
    print(f"close pairs (<{args.close_thr_mm} mm): {close_mask.sum()}/{len(close_mask)} "
          f"({close_mask.mean() * 100:.1f}%)")

    write_summary(out_dir, args, len(xyz_pt), len(xyz_ds), names_pt, names_ds,
                  stats_pt, stats_ds, match, close_mask, thr)
    make_figures(out_dir, xyz_pt, tac_pt, xyz_ds, tac_ds, match, close_mask, thr)


if __name__ == "__main__":
    main()
