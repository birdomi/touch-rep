#!/usr/bin/env python3
"""
visualize_tsne_angle_grasp.py

t-SNE visualization of AngleTransformer CLS (register-token) embeddings
for grasp_prediction dataset.

Compares:
  - scratch  : randomly initialized encoder
  - pretrained: encoder loaded from checkpoints/dinov2_angle/epoch-0040.ckpt

For each model, embeddings are colored by grasp label (success / fail).

Usage:
    python visualize_tsne_angle_grasp.py
    python visualize_tsne_angle_grasp.py --output outputs/tsne_angle_grasp.png
    python visualize_tsne_angle_grasp.py --max_episodes_per_class 50
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b), replace=True)
OmegaConf.register_new_resolver("int_divide",   lambda a, b: a // b,     replace=True)
OmegaConf.register_new_resolver("capitalize",   lambda s: s.title(),     replace=True)

from tactile_ssl.model.angle_transformer import angle_tiny
from tactile_ssl.data.brainco_angle_grasp_dataset import BraincoAngleGraspDataset
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.downstream_task.brainco_grasp_sl import BraincoGraspProbe


# ── model helpers ─────────────────────────────────────────────────────────────

def build_encoder() -> torch.nn.Module:
    return angle_tiny(
        in_dim=10,
        in_chans=4,
        pos_in_dim=10,
        pos_in_chans=4,
        sequence_length=1,
        time_chunk_size=1,
        num_register_tokens=1,
    )


def build_full_model():
    """Build encoder + AttentivePooler + BraincoGraspProbe (same arch as training).

    Matches BraincoGraspDetectionSLModule:
        self.pooler     = AttentivePooler(embed_dim, num_heads)
        self.classifier = BraincoGraspProbe(num_classes=2, ...)
    """
    encoder    = build_encoder()
    embed_dim  = encoder.embed_dim   # 192
    num_heads  = encoder.num_heads   # 3
    pooler     = AttentivePooler(embed_dim=embed_dim, num_heads=num_heads)
    classifier = BraincoGraspProbe(num_classes=2, embed_dim=embed_dim, num_heads=num_heads)
    return encoder, pooler, classifier


def load_full_model_from_ckpt(encoder, pooler, classifier, ckpt_path: str):
    """Load encoder + pooler + classifier from a task last.ckpt.

    Key prefixes in the checkpoint:
        model_encoder.* → encoder
        pooler.*        → pooler  (standalone AttentivePooler used in encode())
        classifier.*    → classifier (BraincoGraspProbe used in forward())
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model") or ckpt.get("state_dict") or ckpt

    def _load(module, prefix):
        raw  = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        msd  = module.state_dict()
        filt = {k: v for k, v in raw.items() if k in msd and v.shape == msd[k].shape}
        module.load_state_dict(filt, strict=False)
        return len(filt), len(raw)

    ne, te = _load(encoder,    "model_encoder.")
    np_, tp = _load(pooler,    "pooler.")
    nc, tc = _load(classifier, "classifier.")
    print(f"  [load] encoder {ne}/{te}  pooler {np_}/{tp}  classifier {nc}/{tc}  ← {ckpt_path}")


def load_encoder_from_ckpt(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    for prefix in ("model_encoder.", "teacher_encoder.backbone.", "student_encoder.backbone."):
        raw = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if raw:
            break
    model_sd = model.state_dict()
    filtered = {k: v for k, v in raw.items() if k in model_sd and v.shape == model_sd[k].shape}
    skipped  = [k for k in raw if k not in filtered]
    if skipped:
        print(f"  [load] Skipped {len(skipped)} shape-mismatched keys: {skipped[:5]}")
    model.load_state_dict(filtered, strict=False)
    print(f"  [load] Loaded {len(filtered)}/{len(raw)} keys from {ckpt_path}")
    return model


# ── val split helpers ─────────────────────────────────────────────────────────

def parse_val_paths_from_split_log(split_log: str) -> set:
    """Parse [val] episode suffixes from a split_foldN.txt file.

    Returns a set of path suffixes like 'grasp_fail/episode_0045_ep0005'.
    """
    val_paths = set()
    with open(split_log) as f:
        for line in f:
            if "[val]" in line:
                # "    [val]   grasp_fail/episode_0045_ep0005"
                val_paths.add(line.split("[val]")[-1].strip())
    return val_paths


def filter_val_episodes(dataset: BraincoAngleGraspDataset, val_suffixes: set) -> list:
    """Return episode_data entries whose path ends with one of val_suffixes."""
    return [
        ep for ep in dataset.episode_data
        if any(ep["path"].endswith(s) for s in val_suffixes)
    ]


# ── embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    encoder: torch.nn.Module,
    dataset: BraincoAngleGraspDataset,
    batch_size: int,
    device: str,
    val_suffixes: Optional[set] = None,
    max_episodes_per_class: Optional[int] = None,
    pooler: Optional[torch.nn.Module] = None,
    classifier: Optional[torch.nn.Module] = None,
) -> tuple:
    """Return (embeddings: np.ndarray [N, D_or_2], labels: np.ndarray [N]).

    embedding_type:
      - pooler=None, classifier=None → encoder CLS token (x_norm_patchtokens mean)
      - pooler only                  → AttentivePooler output
      - pooler + classifier          → pre-probe features (classifier blocks output)
    Each episode: 10 random frames → each frame gives 1 embedding.
    """
    model = encoder
    model.eval()
    model.to(device)
    if pooler is not None:
        pooler.eval(); pooler.to(device)
    if classifier is not None:
        classifier.eval(); classifier.to(device)

    # ── episode selection ─────────────────────────────────────────────────────
    if val_suffixes is not None:
        episodes = filter_val_episodes(dataset, val_suffixes)
    else:
        episodes = list(dataset.episode_data)

    if max_episodes_per_class is not None:
        success_eps = [e for e in episodes if e["label"] == 1][:max_episodes_per_class]
        fail_eps    = [e for e in episodes if e["label"] == 0][:max_episodes_per_class]
        episodes = success_eps + fail_eps

    # ── collect all frames across episodes ────────────────────────────────────
    # Re-read raw frames from windows: each window has W frames; we flatten to
    # individual frames by taking only the FIRST window per episode to avoid
    # duplicates when overlap=0 but still get all frames.
    # Simpler: collect frames from all windows, deduplicate by (episode, frame_idx).
    # Since window_overlap=0 and shift=W, windows are non-overlapping → just flatten.
    selected_paths = {e["path"] for e in episodes}
    ep_windows: dict = {}  # path → sorted list of windows
    for w in dataset.windows:
        if w["episode_path"] not in selected_paths:
            continue
        ep_windows.setdefault(w["episode_path"], []).append(w)

    # 에피소드당 랜덤 10 프레임 샘플 → 10 embeddings/episode
    FRAMES_PER_EP = 10
    rng = np.random.default_rng(42)
    all_embs:   List[torch.Tensor] = []
    all_labels: List[int]          = []
    total_frames = 0

    for ep in episodes:
        path  = ep["path"]
        label = ep["label"]
        wins  = ep_windows.get(path, [])
        if not wins:
            continue

        frames_c = [w["joint_contact"][f]
                    for w in wins for f in range(w["joint_contact"].shape[0])]
        frames_a = [w["finger_angles"][f]
                    for w in wins for f in range(w["finger_angles"].shape[0])]

        n = len(frames_c)
        idxs = rng.choice(n, size=min(FRAMES_PER_EP, n), replace=False)
        total_frames += len(idxs)

        bc = torch.stack([frames_c[i] for i in idxs]).unsqueeze(1).to(device)  # (K, 1, 10, 4)
        ba = torch.stack([frames_a[i] for i in idxs]).unsqueeze(1).to(device)  # (K, 1, 10, 4)
        out = model.forward_features(bc, ba)

        if pooler is not None:
            # x_tokens: (K, reg+N, D) → pooler → (K, 1, D) → squeeze
            x_tokens = out["x_tokens"]                          # (K, reg+N, D)
            pooled   = pooler(x_tokens).squeeze(1)              # (K, D)
            if classifier is not None:
                # BraincoGraspProbe: takes (B, W, D) — use W=1 per frame
                z = pooled.unsqueeze(1)                         # (K, 1, D)
                z = z + classifier.pos_embed_fn(z.device).float().unsqueeze(0)[:, :1, :]
                for block in classifier.blocks:
                    z = block(z, classifier.attn_bias[:, :, :1, :1])
                regs = z[:, 0, :].cpu()                         # (K, D) pre-probe features
            else:
                regs = pooled.cpu()                             # (K, D) pooler output
        else:
            regs = out["x_norm_patchtokens"].mean(dim=1).cpu()  # (K, D) encoder patch mean

        all_embs.append(regs)
        all_labels.extend([label] * len(idxs))

    print(f"  Using {total_frames} frames from {len(episodes)} episodes ({FRAMES_PER_EP} per episode)")

    embeddings = torch.cat(all_embs, dim=0).float().numpy()  # (N_episodes*10, D)
    labels_np  = np.array(all_labels, dtype=np.int32)
    return embeddings, labels_np


# ── t-SNE + plot ──────────────────────────────────────────────────────────────

def run_tsne(embeddings: np.ndarray, perplexity: int = 300) -> np.ndarray:
    n = embeddings.shape[0]
    perplexity = min(perplexity, n - 1)
    print(f"  Running t-SNE on {n} embeddings (dim={embeddings.shape[1]}, perplexity={perplexity}) ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000, init="pca", learning_rate="auto")
    return tsne.fit_transform(embeddings)


def plot_tsne(
    coords_scratch:    np.ndarray,
    labels_scratch:    np.ndarray,
    coords_pretrained: np.ndarray,
    labels_pretrained: np.ndarray,
    output_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE of Pre-probe Features (Encoder + Pooler + Classifier blocks)\n(grasp_prediction val set)",
                 fontsize=13, fontweight="bold")

    style = {
        1: dict(color="#4C9BE8", label="success", marker="o", s=18, alpha=0.7),
        0: dict(color="#E8694C", label="fail",    marker="x", s=22, alpha=0.7, linewidths=0.8),
    }

    for ax, coords, labels, title in [
        (axes[0], coords_scratch,    labels_scratch,    "Scratch (trained from random init)"),
        (axes[1], coords_pretrained, labels_pretrained, "Pretrained (SSL DINOv2-angle init)"),
    ]:
        for lv, st in style.items():
            mask = labels == lv
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1], **st)

        n_success = (labels == 1).sum()
        n_fail    = (labels == 0).sum()
        ax.set_title(f"{title}\n(success={n_success}, fail={n_fail})", fontsize=11)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.legend(fontsize=9, markerscale=1.2)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out.resolve()}")


# ── main ──────────────────────────────────────────────────────────────────────

_EXP_ROOT        = "experiments/angle_grasp_prediction"
_SCRATCH_EXP     = f"{_EXP_ROOT}/2026.05.18_17-33_angle_grasp_prediction_scratch"
_PRETRAINED_EXP  = f"{_EXP_ROOT}/2026.05.19_09-14_angle_grasp_prediction_pretrained"
_SCRATCH_CKPT    = f"{_SCRATCH_EXP}/checkpoints_fold0/last.ckpt"
_PRETRAINED_CKPT = f"{_PRETRAINED_EXP}/checkpoints_fold0/last.ckpt"
_SCRATCH_SPLIT   = f"{_SCRATCH_EXP}/split_fold0.txt"
_PRETRAINED_SPLIT= f"{_PRETRAINED_EXP}/split_fold0.txt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="dataset/brainco/downstream/grasp_prediction")
    p.add_argument("--scratch_ckpt",     default=_SCRATCH_CKPT)
    p.add_argument("--pretrained_ckpt",  default=_PRETRAINED_CKPT)
    p.add_argument("--scratch_split",    default=_SCRATCH_SPLIT,
                   help="split_foldN.txt for scratch val set")
    p.add_argument("--pretrained_split", default=_PRETRAINED_SPLIT,
                   help="split_foldN.txt for pretrained val set")
    p.add_argument("--output", default="outputs/tsne_angle_grasp_val.png")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_episodes_per_class", type=int, default=None)
    p.add_argument("--perplexity", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # ── dataset ───────────────────────────────────────────────────────────────
    data_cfg = OmegaConf.create({
        "window_time":        0.15,
        "window_overlap":     0.0,
        "interpolating_freq": 100,
    })
    print(f"\nLoading dataset from {args.data_path} ...")
    dataset = BraincoAngleGraspDataset(config=data_cfg, data_path=args.data_path)
    print(f"Dataset: {len(dataset.episode_data)} episodes, {len(dataset.windows)} windows")

    # ── val split ─────────────────────────────────────────────────────────────
    scratch_val   = parse_val_paths_from_split_log(args.scratch_split)
    pretrained_val= parse_val_paths_from_split_log(args.pretrained_split)
    print(f"Val episodes — scratch: {len(scratch_val)}, pretrained: {len(pretrained_val)}")

    # ── scratch: full model (encoder + pooler + classifier) ──────────────────
    print(f"\n[1/2] Scratch: {args.scratch_ckpt}")
    enc_s, pool_s, cls_s = build_full_model()
    load_full_model_from_ckpt(enc_s, pool_s, cls_s, args.scratch_ckpt)
    emb_scratch, lbl_scratch = extract_embeddings(
        enc_s, dataset, args.batch_size, args.device,
        val_suffixes=scratch_val,
        max_episodes_per_class=args.max_episodes_per_class,
        pooler=pool_s, classifier=cls_s,
    )
    coords_scratch = run_tsne(emb_scratch, perplexity=args.perplexity)

    # ── pretrained: full model ────────────────────────────────────────────────
    print(f"\n[2/2] Pretrained: {args.pretrained_ckpt}")
    enc_p, pool_p, cls_p = build_full_model()
    load_full_model_from_ckpt(enc_p, pool_p, cls_p, args.pretrained_ckpt)
    emb_pre, lbl_pre = extract_embeddings(
        enc_p, dataset, args.batch_size, args.device,
        val_suffixes=pretrained_val,
        max_episodes_per_class=args.max_episodes_per_class,
        pooler=pool_p, classifier=cls_p,
    )
    coords_pre = run_tsne(emb_pre, perplexity=args.perplexity)

    # ── plot ──────────────────────────────────────────────────────────────────
    print("\nPlotting ...")
    plot_tsne(coords_scratch, lbl_scratch, coords_pre, lbl_pre, args.output)


if __name__ == "__main__":
    main()
