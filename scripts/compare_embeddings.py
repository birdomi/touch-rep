"""Compare data sources in tactile-encoder embedding space.

Embeds per-frame (T=1) tactile+fingertip-xyz inputs from:
  - BrainCo SSL pretraining data      (pretraining_dataset/brainco)
  - slip detection data               (dataset/brainco/downstream/slip_data_v2)
  - grasp prediction data (reference) (dataset/brainco/downstream/grasp_prediction)

with a pretrained XYZ encoder (default: the `rope` checkpoint
dinov2_all_pseudo_force_tiny_rope_v2_hp1 epoch-0100, teacher weights) and
compares the embedding distributions:
  - PCA / t-SNE 2D projections by source (+ slip frames by slip label)
  - NN cosine distance: downstream -> pretrain vs pretrain -> pretrain (LOO)
  - centroid cosine similarity, kNN domain purity
  - slip-label separability via LOO kNN in embedding space

Raw arrays are reused from the compare_pretrain_downstream_raw.py caches
(same episode selection / strides); missing caches are rebuilt automatically.

Usage:
  XFORMERS_DISABLED=TRUE python scripts/compare_embeddings.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf  # noqa: E402
import hydra  # noqa: E402

from tactile_ssl.data.force_channels import force_direction_to_cartesian  # noqa: E402
from tactile_ssl.data.brainco_xyz_grasp_dataset import _align_xyz_to_npz  # noqa: E402
from tactile_ssl.data.brainco_xyz_slip_detection_dataset import _read_slip_labels  # noqa: E402

from compare_pretrain_downstream_raw import (  # noqa: E402
    iter_pretrain_episodes, iter_downstream_episodes, pick_spread, load_source,
)

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b, replace=True)
OmegaConf.register_new_resolver("capitalize", lambda s: s.title(), replace=True)

MAX_VALUES = np.array([1000, 1000, 1000, 100000], dtype=np.float32)

# Okabe-Ito, consistent with the raw comparison figures
C_PT = "#0072B2"    # pretrain
C_SLIP = "#E69F00"  # slip
C_GRASP = "#009E73" # grasp
C_SLIP_POS = "#D55E00"  # slip label = slip
C_SLIP_NEG = "#999999"  # slip label = non-slip

DEFAULT_CKPT = ("experiments/dinov2_all_pseudo_force_tiny_rope_v2_hp1/"
                "2026.07.27-22-14/checkpoints/epoch-0100.ckpt")


# ── Encoder ──────────────────────────────────────────────────────────────────

def build_encoder(ckpt_path: Path, device: str):
    run_dir = ckpt_path.parent.parent
    cfg = OmegaConf.load(run_dir / "config.yaml")
    enc_cfg = OmegaConf.to_container(cfg.algorithm.encoder, resolve=True)
    if enc_cfg.get("normalization") and enc_cfg["normalization"].get("mean") is None:
        # signal_mean/std buffers are restored from the checkpoint below
        enc_cfg["normalization"] = {"mean": [0.0] * 4, "std": [1.0] * 4}
    encoder = hydra.utils.instantiate(enc_cfg)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    prefix = "teacher_encoder.backbone."
    enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    print(f"[encoder] loaded {len(enc_state)} tensors from {ckpt_path.name} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"[encoder] missing keys (first 5): {missing[:5]}")
    encoder.eval().to(device)
    return encoder


def preprocess(tac_raw: np.ndarray, xyz_raw: np.ndarray):
    """Raw cache arrays -> encoder inputs, mirroring BraincoXYZ*Dataset.

    tac_raw (F,10,4) raw channels [Fn, Ft, dir(-1 invalid), prox]
    xyz_raw (F,10,3) fingertip_rel (BrainCo wrist-local)
    Returns torch tensors: sensor (F,10,4), pos (F,10,3)
    """
    converted, valid = force_direction_to_cartesian(tac_raw)
    converted = np.where(valid, converted, 0.0).astype(np.float32)
    converted = converted / MAX_VALUES.reshape(1, 1, -1)
    sensor = torch.from_numpy(converted)
    pos = _align_xyz_to_npz(torch.from_numpy(xyz_raw.astype(np.float32)))
    return sensor, pos


@torch.inference_mode()
def embed(encoder, sensor, pos, batch_size, device):
    outs = []
    for s in range(0, len(sensor), batch_size):
        x = sensor[s:s + batch_size].to(device).unsqueeze(1)   # (B,1,10,4)
        p = pos[s:s + batch_size].to(device).unsqueeze(1)      # (B,1,10,3)
        feats = encoder.forward_features(x, p)
        emb = feats["x_norm_regtokens"].mean(dim=1)            # (B, D)
        outs.append(emb.float().cpu())
    return torch.cat(outs)


# ── Metrics ──────────────────────────────────────────────────────────────────

def nn_cos_dist(query: torch.Tensor, ref: torch.Tensor, exclude_self=False, chunk=2048):
    """Min cosine distance from each L2-normalized query row to ref rows."""
    dists = []
    for s in range(0, len(query), chunk):
        sim = query[s:s + chunk] @ ref.T
        if exclude_self:
            idx = torch.arange(s, min(s + chunk, len(query)))
            sim[torch.arange(len(idx)), idx] = -1.0
        dists.append(1.0 - sim.max(dim=1).values)
    return torch.cat(dists).numpy()


def knn_domain_purity(emb_a: torch.Tensor, emb_b: torch.Tensor, k=10, n_sub=1500, seed=0):
    """Balanced kNN two-sample statistic: mean fraction of same-domain neighbours.

    0.5 = indistinguishable domains, 1.0 = fully separated.
    """
    g = torch.Generator().manual_seed(seed)
    a = emb_a[torch.randperm(len(emb_a), generator=g)[:n_sub]]
    b = emb_b[torch.randperm(len(emb_b), generator=g)[:n_sub]]
    x = torch.cat([a, b])
    y = torch.cat([torch.zeros(len(a)), torch.ones(len(b))])
    sim = x @ x.T
    sim.fill_diagonal_(-1.0)
    nn_idx = sim.topk(k, dim=1).indices
    same = (y[nn_idx] == y[:, None]).float().mean().item()
    return same


def slip_label_knn(emb: torch.Tensor, labels: np.ndarray, k=10, episode_ids=None):
    """kNN balanced accuracy for slip labels within slip embeddings.

    episode_ids=None: leave-one-out (inflated by within-episode temporal
    correlation). With episode_ids, neighbours from the query's own episode
    are excluded (cross-episode generalization).
    """
    y = torch.from_numpy(labels.astype(np.int64))
    sim = emb @ emb.T
    if episode_ids is None:
        sim.fill_diagonal_(-1.0)
    else:
        ep = torch.from_numpy(episode_ids)
        sim[ep[:, None] == ep[None, :]] = -1.0
    nn_idx = sim.topk(k, dim=1).indices
    pred = (y[nn_idx].float().mean(dim=1) > 0.5).long()
    accs = [(pred[y == c] == c).float().mean().item() for c in (0, 1)]
    return float(np.mean(accs)), accs


# ── Slip labels aligned to the cached frame order ────────────────────────────

def slip_labels_for_cache(episodes, stride):
    labels, ep_ids = [], []
    for ep_idx, (_, ep_path) in enumerate(episodes):
        with open(ep_path / "data.json") as f:
            n = len(json.load(f)["data"])
        lab = _read_slip_labels(ep_path, n)[::stride]
        labels.append(lab)
        ep_ids.append(np.full(len(lab), ep_idx, dtype=np.int64))
    return np.concatenate(labels), np.concatenate(ep_ids)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--pretrain_root", default="pretraining_dataset/brainco")
    ap.add_argument("--slip_root", default="dataset/brainco/downstream/slip_data_v2")
    ap.add_argument("--grasp_root", default="dataset/brainco/downstream/grasp_prediction")
    ap.add_argument("--max_episodes_pretrain", type=int, default=12)
    ap.add_argument("--frame_stride_pretrain", type=int, default=10)
    ap.add_argument("--max_episodes_slip", type=int, default=24)
    ap.add_argument("--frame_stride_slip", type=int, default=2)
    ap.add_argument("--max_episodes_grasp", type=int, default=30)
    ap.add_argument("--frame_stride_grasp", type=int, default=1)
    ap.add_argument("--urdf", default="dataset/brainco/urdf")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tsne", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--raw_cache_pretrain", default="scripts/compare_raw_out")
    ap.add_argument("--raw_cache_slip", default="scripts/compare_raw_out_slip")
    ap.add_argument("--raw_cache_grasp", default="scripts/compare_raw_out")
    ap.add_argument("--out", default="scripts/compare_emb_out")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    urdf = Path(args.urdf)

    # ── Load raw arrays (cache hits if the raw comparison already ran) ──
    eps_pt = pick_spread(iter_pretrain_episodes(Path(args.pretrain_root)), args.max_episodes_pretrain)
    eps_slip = pick_spread(iter_downstream_episodes(Path(args.slip_root)), args.max_episodes_slip)
    eps_grasp = pick_spread(iter_downstream_episodes(Path(args.grasp_root)), args.max_episodes_grasp)

    xyz_pt, tac_pt, _ = load_source("pretrain", eps_pt, urdf,
                                    args.frame_stride_pretrain, Path(args.raw_cache_pretrain))
    xyz_sl, tac_sl, _ = load_source("downstream", eps_slip, urdf,
                                    args.frame_stride_slip, Path(args.raw_cache_slip))
    xyz_gr, tac_gr, _ = load_source("downstream", eps_grasp, urdf,
                                    args.frame_stride_grasp, Path(args.raw_cache_grasp))

    slip_labels, slip_ep_ids = slip_labels_for_cache(eps_slip, args.frame_stride_slip)
    assert len(slip_labels) == len(xyz_sl), (len(slip_labels), len(xyz_sl))

    # ── Embed ──
    encoder = build_encoder(Path(args.checkpoint), args.device)
    embs = {}
    for name, (tac, xyz) in {
        "pretrain": (tac_pt, xyz_pt),
        "slip": (tac_sl, xyz_sl),
        "grasp": (tac_gr, xyz_gr),
    }.items():
        sensor, pos = preprocess(tac, xyz)
        e = embed(encoder, sensor, pos, args.batch_size, args.device)
        embs[name] = F.normalize(e, dim=-1)
        print(f"[embed] {name}: {tuple(e.shape)}")

    np.savez_compressed(out_dir / "embeddings.npz",
                        **{k: v.numpy() for k, v in embs.items()},
                        slip_labels=slip_labels)

    # ── Metrics ──
    cent = {k: F.normalize(v.mean(dim=0), dim=-1) for k, v in embs.items()}
    cent_sim = {
        "pretrain-slip": float(cent["pretrain"] @ cent["slip"]),
        "pretrain-grasp": float(cent["pretrain"] @ cent["grasp"]),
        "slip-grasp": float(cent["slip"] @ cent["grasp"]),
    }
    d_self = nn_cos_dist(embs["pretrain"], embs["pretrain"], exclude_self=True)
    d_slip = nn_cos_dist(embs["slip"], embs["pretrain"])
    d_grasp = nn_cos_dist(embs["grasp"], embs["pretrain"])
    purity_slip = knn_domain_purity(embs["pretrain"], embs["slip"])
    purity_grasp = knn_domain_purity(embs["pretrain"], embs["grasp"])
    slip_bal_acc, slip_accs = slip_label_knn(embs["slip"], slip_labels)
    slip_bal_acc_xep, slip_accs_xep = slip_label_knn(
        embs["slip"], slip_labels, episode_ids=slip_ep_ids
    )

    # ── Summary ──
    def q(d):
        return f"p50={np.median(d):.4f}, p95={np.percentile(d, 95):.4f}, max={d.max():.4f}"

    md = [
        "# Embedding-space comparison (tactile encoder)",
        "",
        f"- encoder: `{args.checkpoint}` (teacher backbone, reg-token mean, L2-normalized)",
        f"- pretrain: {len(embs['pretrain'])} frames | slip: {len(embs['slip'])} frames "
        f"(slip label ratio {slip_labels.mean():.2f}) | grasp: {len(embs['grasp'])} frames",
        "",
        "## Centroid cosine similarity",
        "",
        *[f"- {k}: {v:.4f}" for k, v in cent_sim.items()],
        "",
        "## NN cosine distance to pretrain set",
        "",
        f"- pretrain -> pretrain (LOO, baseline): {q(d_self)}",
        f"- slip -> pretrain: {q(d_slip)}",
        f"- grasp -> pretrain: {q(d_grasp)}",
        "",
        "## kNN domain purity vs pretrain (0.5 = indistinguishable, 1.0 = separated)",
        "",
        f"- slip vs pretrain: {purity_slip:.3f}",
        f"- grasp vs pretrain: {purity_grasp:.3f}",
        "",
        "## Slip-label separability inside slip embeddings (LOO kNN, k=10)",
        "",
        f"- LOO (within-episode leakage possible): balanced acc {slip_bal_acc:.3f} "
        f"(non-slip recall {slip_accs[0]:.3f}, slip recall {slip_accs[1]:.3f})",
        f"- cross-episode (own episode excluded): balanced acc {slip_bal_acc_xep:.3f} "
        f"(non-slip recall {slip_accs_xep[0]:.3f}, slip recall {slip_accs_xep[1]:.3f})",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    print(f"wrote {out_dir / 'summary.md'}")

    # ── Figures ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_emb = torch.cat([embs["pretrain"], embs["slip"], embs["grasp"]]).numpy()
    src = np.concatenate([
        np.zeros(len(embs["pretrain"])), np.ones(len(embs["slip"])),
        np.full(len(embs["grasp"]), 2),
    ])

    from sklearn.decomposition import PCA
    p2 = PCA(n_components=2, random_state=0).fit_transform(all_emb)
    projs = [("PCA", p2)]
    if args.tsne:
        from sklearn.manifold import TSNE
        print("running t-SNE ...")
        t2 = TSNE(n_components=2, random_state=0, init="pca",
                  perplexity=30).fit_transform(all_emb)
        projs.append(("t-SNE", t2))

    for pname, pts in projs:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
        ax = axes[0]
        for si, (label, color) in enumerate(
            [("pretrain", C_PT), ("slip", C_SLIP), ("grasp", C_GRASP)]
        ):
            m = src == si
            ax.scatter(pts[m, 0], pts[m, 1], s=4, c=color, alpha=0.35, lw=0, label=label)
        leg = ax.legend(fontsize=9, markerscale=4)
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
        ax.set_title(f"{pname} — by source")
        ax.grid(alpha=0.25)

        ax = axes[1]
        m_pt = src == 0
        ax.scatter(pts[m_pt, 0], pts[m_pt, 1], s=4, c="#CCCCCC", alpha=0.3, lw=0, label="pretrain")
        m_slip = src == 1
        sl_pts = pts[m_slip]
        for lab, color, name in [(0, C_SLIP_NEG, "non-slip"), (1, C_SLIP_POS, "slip")]:
            m = slip_labels == lab
            ax.scatter(sl_pts[m, 0], sl_pts[m, 1], s=4, c=color, alpha=0.45, lw=0, label=name)
        leg = ax.legend(fontsize=9, markerscale=4)
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
        ax.set_title(f"{pname} — slip frames by slip label")
        ax.grid(alpha=0.25)
        fig.suptitle(f"Tactile encoder embeddings ({pname})")
        fname = f"fig_{pname.lower().replace('-', '')}.png"
        fig.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.linspace(0, max(d_slip.max(), d_grasp.max(), d_self.max()), 80)
    ax.hist(d_self, bins=bins, density=True, histtype="step", lw=2, color=C_PT,
            label="pretrain->pretrain (LOO)")
    ax.hist(d_grasp, bins=bins, density=True, histtype="step", lw=2, color=C_GRASP,
            label="grasp->pretrain")
    ax.hist(d_slip, bins=bins, density=True, histtype="step", lw=2, color=C_SLIP,
            label="slip->pretrain")
    ax.set_yscale("log")
    ax.set_xlabel("NN cosine distance")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title("Embedding NN distance to the pretraining set")
    fig.savefig(out_dir / "fig_nn_cosdist.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
