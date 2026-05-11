"""analyze_coord_alignment.py — Arctic / HOT3D / OakInkV2 / TACO 좌표계 정렬 분석.

각 데이터셋에서 샘플을 추출하여:
  1. 전역(vroot) 손목 위치/회전 분포
  2. 로컬 손 관절 위치 분포
  3. 접촉(tactile) 값 분포
  4. 좌우 대칭성 (LH x < 0, RH x > 0 비율)
를 통계로 출력하고, 시각화 이미지를 저장.

Usage:
    python analyze_coord_alignment.py --out vis/coord_alignment.png
"""
import argparse
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tactile_ssl.data.oakinkv2_tactile import OakInkV2TactileDataset

_MP_TIPS = [4, 8, 12, 16, 20]
_MP_LINES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

DATASETS = {
    "Arctic":  "pretraining_dataset/Arctic",
    "HOT3D":   "pretraining_dataset/hot3d",
    "OakInkV2":"pretraining_dataset/OakInkv2",
    "TACO":    "pretraining_dataset/TACO",
}
COLORS = {
    "Arctic":   "#E57373",
    "HOT3D":    "#64B5F6",
    "OakInkV2": "#81C784",
    "TACO":     "#FFB74D",
}


# ── 데이터 수집 ────────────────────────────────────────────────────────────────

def collect_stats(data_root: str, n_seq: int, window_size: int, seed: int) -> dict:
    """데이터셋에서 n_seq개 시퀀스를 샘플링하여 통계 배열 반환."""
    ds = OakInkV2TactileDataset(
        data_root=data_root,
        window_size=window_size,
        window_stride=window_size,   # non-overlapping
        split="train",
        train_val_split=1.0,
    )
    n = len(ds)
    rng = random.Random(seed)
    indices = rng.sample(range(n), min(n_seq, n))

    lh_wrist_pos  = []   # (N, 3)  LH wrist position in vroot
    rh_wrist_pos  = []   # (N, 3)  RH wrist position in vroot
    lh_wrist_rot  = []   # (N, 6)  LH wrist rot6d in vroot
    rh_wrist_rot  = []   # (N, 6)  RH wrist rot6d in vroot
    lh_joints_loc = []   # (N, 21, 3)  LH joints in local wrist frame
    rh_joints_loc = []   # (N, 21, 3)  RH joints in local wrist frame
    lh_contact    = []   # (N,)  LH fingertip contact sum
    rh_contact    = []   # (N,)  RH fingertip contact sum

    for idx in indices:
        sample = ds[idx]
        # frame 0 사용
        poses   = sample["sensor_poses"][0].numpy()   # (42, 3)
        wrists  = sample["wrist_poses"][0].numpy()    # (2, 9)
        sensor  = sample["sensor"][0].numpy()         # (42, 1)

        lh_wrist_pos.append(wrists[0, :3])
        rh_wrist_pos.append(wrists[1, :3])
        lh_wrist_rot.append(wrists[0, 3:])
        rh_wrist_rot.append(wrists[1, 3:])
        lh_joints_loc.append(poses[:21])
        rh_joints_loc.append(poses[21:])
        lh_contact.append(sensor[_MP_TIPS, 0].sum())
        rh_contact.append(sensor[[21+t for t in _MP_TIPS], 0].sum())

    return dict(
        lh_wrist_pos  = np.stack(lh_wrist_pos),
        rh_wrist_pos  = np.stack(rh_wrist_pos),
        lh_wrist_rot  = np.stack(lh_wrist_rot),
        rh_wrist_rot  = np.stack(rh_wrist_rot),
        lh_joints_loc = np.stack(lh_joints_loc),
        rh_joints_loc = np.stack(rh_joints_loc),
        lh_contact    = np.array(lh_contact),
        rh_contact    = np.array(rh_contact),
        n             = len(indices),
    )


# ── 통계 출력 ─────────────────────────────────────────────────────────────────

def print_stats(name: str, stats: dict):
    lhp = stats["lh_wrist_pos"]   # (N, 3)
    rhp = stats["rh_wrist_pos"]
    ljl = stats["lh_joints_loc"]  # (N, 21, 3)
    rjl = stats["rj_joints_loc"] if "rj_joints_loc" in stats else stats["rh_joints_loc"]

    # 좌우 대칭성: LH x < 0, RH x > 0 인 비율
    lh_neg_x = (lhp[:, 0] < 0).mean() * 100
    rh_pos_x = (rhp[:, 0] > 0).mean() * 100
    # 손목 간 거리
    dist = np.linalg.norm(rhp - lhp, axis=-1)

    print(f"\n{'='*60}")
    print(f"  {name}  (N={stats['n']})")
    print(f"{'='*60}")
    print(f"  [LH wrist pos (vroot)]  mean={lhp.mean(0)}, std={lhp.std(0)}")
    print(f"  [RH wrist pos (vroot)]  mean={rhp.mean(0)}, std={rhp.std(0)}")
    print(f"  [LH x < 0]  {lh_neg_x:.1f}%   [RH x > 0]  {rh_pos_x:.1f}%")
    print(f"  [LH-RH dist]  mean={dist.mean():.4f}  std={dist.std():.4f}  "
          f"min={dist.min():.4f}  max={dist.max():.4f}")

    # 로컬 관절 범위
    lj_flat = ljl.reshape(-1, 3)
    rj_flat = stats["rh_joints_loc"].reshape(-1, 3)
    print(f"  [LH local joints]  "
          f"x[{lj_flat[:,0].min():.3f}, {lj_flat[:,0].max():.3f}]  "
          f"y[{lj_flat[:,1].min():.3f}, {lj_flat[:,1].max():.3f}]  "
          f"z[{lj_flat[:,2].min():.3f}, {lj_flat[:,2].max():.3f}]")
    print(f"  [RH local joints]  "
          f"x[{rj_flat[:,0].min():.3f}, {rj_flat[:,0].max():.3f}]  "
          f"y[{rj_flat[:,1].min():.3f}, {rj_flat[:,1].max():.3f}]  "
          f"z[{rj_flat[:,2].min():.3f}, {rj_flat[:,2].max():.3f}]")

    # 접촉 통계
    lhc, rhc = stats["lh_contact"], stats["rh_contact"]
    print(f"  [Contact]  LH mean={lhc.mean():.4f}  RH mean={rhc.mean():.4f}  "
          f"any_contact={((lhc+rhc)>0).mean()*100:.1f}%")


# ── 시각화 ────────────────────────────────────────────────────────────────────

def _rot6d_to_mat(r6d: np.ndarray) -> np.ndarray:
    c0 = r6d[..., :3]; c1 = r6d[..., 3:]
    c0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1 = c1 - (c1 * c0).sum(axis=-1, keepdims=True) * c0
    c1 = c1 / (np.linalg.norm(c1, axis=-1, keepdims=True) + 1e-8)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=-1)


def make_figure(all_stats: dict, out_path: str):
    names = list(all_stats.keys())
    colors = [COLORS[n] for n in names]

    fig = plt.figure(figsize=(22, 26), constrained_layout=True)
    fig.suptitle("Pretraining Dataset Coordinate Alignment Analysis", fontsize=13, y=1.01)

    gs = fig.add_gridspec(5, 4)

    # ── Row 0: LH & RH wrist X position (vroot) ───────────────────────────────
    ax_lhx = fig.add_subplot(gs[0, 0])
    ax_rhx = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_lhx_y = fig.add_subplot(gs[0, 3])

    for name, color in zip(names, colors):
        s = all_stats[name]
        lhx = s["lh_wrist_pos"][:, 0]
        rhx = s["rh_wrist_pos"][:, 0]
        dist = np.linalg.norm(s["rh_wrist_pos"] - s["lh_wrist_pos"], axis=-1)
        lhy = s["lh_wrist_pos"][:, 1]
        ax_lhx.hist(lhx, bins=50, alpha=0.55, color=color, label=name, density=True)
        ax_rhx.hist(rhx, bins=50, alpha=0.55, color=color, label=name, density=True)
        ax_dist.hist(dist, bins=50, alpha=0.55, color=color, label=name, density=True)
        ax_lhx_y.hist(lhy, bins=50, alpha=0.55, color=color, label=name, density=True)

    ax_lhx.axvline(0, color="black", lw=1.5, ls="--")
    ax_rhx.axvline(0, color="black", lw=1.5, ls="--")
    ax_lhx.set_title("LH wrist X (vroot)\n← should be < 0")
    ax_rhx.set_title("RH wrist X (vroot)\n← should be > 0")
    ax_dist.set_title("LH-RH wrist distance (m)")
    ax_lhx_y.set_title("LH wrist Y (vroot)")
    for ax in [ax_lhx, ax_rhx, ax_dist, ax_lhx_y]:
        ax.legend(fontsize=7); ax.set_ylabel("density")

    # ── Row 1: LH & RH wrist Y, Z position ────────────────────────────────────
    ax_lhy = fig.add_subplot(gs[1, 0])
    ax_lhz = fig.add_subplot(gs[1, 1])
    ax_rhy = fig.add_subplot(gs[1, 2])
    ax_rhz = fig.add_subplot(gs[1, 3])

    for name, color in zip(names, colors):
        s = all_stats[name]
        ax_lhy.hist(s["lh_wrist_pos"][:, 1], bins=50, alpha=0.55, color=color, label=name, density=True)
        ax_lhz.hist(s["lh_wrist_pos"][:, 2], bins=50, alpha=0.55, color=color, label=name, density=True)
        ax_rhy.hist(s["rh_wrist_pos"][:, 1], bins=50, alpha=0.55, color=color, label=name, density=True)
        ax_rhz.hist(s["rh_wrist_pos"][:, 2], bins=50, alpha=0.55, color=color, label=name, density=True)

    ax_lhy.set_title("LH wrist Y (vroot)"); ax_lhz.set_title("LH wrist Z (vroot)")
    ax_rhy.set_title("RH wrist Y (vroot)"); ax_rhz.set_title("RH wrist Z (vroot)")
    for ax in [ax_lhy, ax_lhz, ax_rhy, ax_rhz]:
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.legend(fontsize=7); ax.set_ylabel("density")

    # ── Row 2: 로컬 관절 분포 (XYZ per axis) ──────────────────────────────────
    ax_jx = fig.add_subplot(gs[2, 0])
    ax_jy = fig.add_subplot(gs[2, 1])
    ax_jz = fig.add_subplot(gs[2, 2])
    ax_ct = fig.add_subplot(gs[2, 3])

    for name, color in zip(names, colors):
        s = all_stats[name]
        lj = s["lh_joints_loc"].reshape(-1, 3)
        rj = s["rh_joints_loc"].reshape(-1, 3)
        all_j = np.concatenate([lj, rj], axis=0)
        ax_jx.hist(all_j[:, 0], bins=60, alpha=0.45, color=color, label=name, density=True)
        ax_jy.hist(all_j[:, 1], bins=60, alpha=0.45, color=color, label=name, density=True)
        ax_jz.hist(all_j[:, 2], bins=60, alpha=0.45, color=color, label=name, density=True)
        contact = np.concatenate([s["lh_contact"], s["rh_contact"]])
        ax_ct.hist(contact[contact > 0], bins=60, alpha=0.45, color=color, label=name, density=True)

    ax_jx.set_title("Joint X (local wrist frame)"); ax_jy.set_title("Joint Y (local)")
    ax_jz.set_title("Joint Z (local)"); ax_ct.set_title("Contact value > 0 (log scale)")
    ax_ct.set_yscale("log")
    for ax in [ax_jx, ax_jy, ax_jz, ax_ct]:
        ax.legend(fontsize=7); ax.set_ylabel("density")

    # ── Row 3: 손목 위치 2D scatter — XY plane ────────────────────────────────
    ax_xy_lh = fig.add_subplot(gs[3, 0])
    ax_xy_rh = fig.add_subplot(gs[3, 1])
    ax_xz_lh = fig.add_subplot(gs[3, 2])
    ax_xz_rh = fig.add_subplot(gs[3, 3])

    for name, color in zip(names, colors):
        s = all_stats[name]
        lhp = s["lh_wrist_pos"]
        rhp = s["rh_wrist_pos"]
        kw = dict(s=4, alpha=0.3, color=color, label=name)
        ax_xy_lh.scatter(lhp[:, 0], lhp[:, 1], **kw)
        ax_xy_rh.scatter(rhp[:, 0], rhp[:, 1], **kw)
        ax_xz_lh.scatter(lhp[:, 0], lhp[:, 2], **kw)
        ax_xz_rh.scatter(rhp[:, 0], rhp[:, 2], **kw)

    for ax, title in [
        (ax_xy_lh, "LH wrist: X-Y"), (ax_xy_rh, "RH wrist: X-Y"),
        (ax_xz_lh, "LH wrist: X-Z"), (ax_xz_rh, "RH wrist: X-Z"),
    ]:
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(title); ax.legend(fontsize=7, markerscale=3)
        ax.set_aspect("equal", adjustable="datalim")

    # ── Row 4: 평균 스켈레톤 overlay ──────────────────────────────────────────
    ax_sk_lh = fig.add_subplot(gs[4, 0], projection="3d")
    ax_sk_rh = fig.add_subplot(gs[4, 1], projection="3d")
    ax_sk_lh_xy = fig.add_subplot(gs[4, 2])
    ax_sk_rh_xy = fig.add_subplot(gs[4, 3])

    for name, color in zip(names, colors):
        s = all_stats[name]
        lj_mean = s["lh_joints_loc"].mean(0)   # (21, 3)
        rj_mean = s["rh_joints_loc"].mean(0)

        for sk, ax3, ax2 in [(lj_mean, ax_sk_lh, ax_sk_lh_xy),
                             (rj_mean, ax_sk_rh, ax_sk_rh_xy)]:
            for i, j in _MP_LINES:
                ax3.plot([sk[i,0], sk[j,0]], [sk[i,1], sk[j,1]], [sk[i,2], sk[j,2]],
                         c=color, lw=1.5, alpha=0.8, label=name if (i, j) == _MP_LINES[0] else None)
                ax2.plot([sk[i,0], sk[j,0]], [sk[i,1], sk[j,1]],
                         c=color, lw=1.5, alpha=0.8, label=name if (i, j) == _MP_LINES[0] else None)
            ax3.scatter(sk[:,0], sk[:,1], sk[:,2], c=color, s=10)
            ax2.scatter(sk[:,0], sk[:,1], c=color, s=8)

    for ax3, title in [(ax_sk_lh, "Mean LH skeleton (local)"),
                       (ax_sk_rh, "Mean RH skeleton (local)")]:
        ax3.set_title(title, fontsize=9)
        ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
        ax3.legend(fontsize=7)

    for ax2, title in [(ax_sk_lh_xy, "Mean LH skeleton XY"),
                       (ax_sk_rh_xy, "Mean RH skeleton XY")]:
        ax2.set_title(title, fontsize=9)
        ax2.set_xlabel("X"); ax2.set_ylabel("Y")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


# ── 정렬 요약 ─────────────────────────────────────────────────────────────────

def print_alignment_summary(all_stats: dict):
    print("\n" + "="*70)
    print("  ALIGNMENT SUMMARY")
    print("="*70)
    header = f"  {'Dataset':12s}  {'LH x<0':>8}  {'RH x>0':>8}  "
    header += f"{'LH x mean':>10}  {'RH x mean':>10}  {'wrist dist':>10}  {'any contact':>12}"
    print(header)
    print("  " + "-"*65)

    for name, s in all_stats.items():
        lhp = s["lh_wrist_pos"]
        rhp = s["rh_wrist_pos"]
        dist = np.linalg.norm(rhp - lhp, axis=-1)
        lh_neg = (lhp[:, 0] < 0).mean() * 100
        rh_pos = (rhp[:, 0] > 0).mean() * 100
        any_ct = ((s["lh_contact"] + s["rh_contact"]) > 0).mean() * 100
        print(f"  {name:12s}  {lh_neg:7.1f}%  {rh_pos:7.1f}%  "
              f"{lhp[:,0].mean():+10.4f}  {rhp[:,0].mean():+10.4f}  "
              f"{dist.mean():10.4f}m  {any_ct:11.1f}%")

    print()
    print("  ※ LH x<0 / RH x>0 가 100%에 가까울수록 좌표계 부호 정렬 일치")
    print("  ※ LH x mean ≈ -RH x mean 이면 좌우 대칭 정렬")

    # 축별 wrist 위치 통계
    print()
    print(f"  {'Dataset':12s}  {'LH x':>12}  {'LH y':>12}  {'LH z':>12}")
    print("  " + "-"*52)
    for name, s in all_stats.items():
        lhp = s["lh_wrist_pos"]
        print(f"  {name:12s}  "
              f"{lhp[:,0].mean():+.3f}±{lhp[:,0].std():.3f}  "
              f"{lhp[:,1].mean():+.3f}±{lhp[:,1].std():.3f}  "
              f"{lhp[:,2].mean():+.3f}±{lhp[:,2].std():.3f}")

    print()
    print(f"  {'Dataset':12s}  {'RH x':>12}  {'RH y':>12}  {'RH z':>12}")
    print("  " + "-"*52)
    for name, s in all_stats.items():
        rhp = s["rh_wrist_pos"]
        print(f"  {name:12s}  "
              f"{rhp[:,0].mean():+.3f}±{rhp[:,0].std():.3f}  "
              f"{rhp[:,1].mean():+.3f}±{rhp[:,1].std():.3f}  "
              f"{rhp[:,2].mean():+.3f}±{rhp[:,2].std():.3f}")

    # 로컬 관절 스케일
    print()
    print(f"  {'Dataset':12s}  {'joint X range':>18}  {'joint Y range':>18}  {'joint Z range':>18}")
    print("  " + "-"*70)
    for name, s in all_stats.items():
        all_j = np.concatenate([
            s["lh_joints_loc"].reshape(-1, 3),
            s["rh_joints_loc"].reshape(-1, 3),
        ], axis=0)
        def rng(col): return f"[{all_j[:,col].min():+.3f}, {all_j[:,col].max():+.3f}]"
        print(f"  {name:12s}  {rng(0):>18}  {rng(1):>18}  {rng(2):>18}")

    # 접촉 통계
    print()
    print(f"  {'Dataset':12s}  {'contact mean(LH)':>17}  {'contact mean(RH)':>17}  {'std(LH)':>10}  {'std(RH)':>10}")
    print("  " + "-"*75)
    for name, s in all_stats.items():
        lhc, rhc = s["lh_contact"], s["rh_contact"]
        print(f"  {name:12s}  {lhc.mean():17.5f}  {rhc.mean():17.5f}  {lhc.std():10.5f}  {rhc.std():10.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seq",       type=int, default=2000,
                        help="데이터셋당 샘플 수 (많을수록 정확)")
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out",         default="vis/coord_alignment.png")
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    all_stats = {}
    for name, root in DATASETS.items():
        if not Path(root).exists():
            print(f"[SKIP] {name}: {root} 없음")
            continue
        print(f"\nLoading {name} from {root} ...")
        all_stats[name] = collect_stats(
            data_root=root,
            n_seq=args.n_seq,
            window_size=args.window_size,
            seed=args.seed,
        )
        print(f"  → {all_stats[name]['n']} windows loaded")

    print_alignment_summary(all_stats)
    make_figure(all_stats, args.out)


if __name__ == "__main__":
    main()
