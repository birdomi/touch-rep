"""
Extract MANO hand joint positions (wrist-local frame) with per-joint contact ratio.

For every mocap frame, computes 21 MANO joints for both hands in world
coordinates, converts them into a wrist-local coordinate frame, and attaches a
per-joint contact ratio derived from contact_anno.

Output shape per frame per hand: (21, 4)  →  [x_local, y_local, z_local, contact_ratio]

Joint index mapping (manotorch, 21 joints):
    0        : wrist
    1- 4     : thumb  (CMC, MCP, IP, tip)
    5- 8     : index  (MCP, PIP, DIP, tip)
    9-12     : middle (MCP, PIP, DIP, tip)
    13-16    : ring   (MCP, PIP, DIP, tip)
    17-20    : pinky  (MCP, PIP, DIP, tip)

Wrist-local axes per hand:
    origin : wrist
    x-axis : back of hand -> palm
    y-axis : pinky_MCP -> thumb_CMC
    z-axis : wrist -> middle_MCP

contact_ratio per joint:
    non-finger tip : fraction of that joint's skinning-dominant vertices in contact
    finger tip:  fraction of the K=20 nearest rest-pose vertices to the
                  fingertip that are in contact

Output file: hand_joint_anno/<seq_name>.pkl
    {
        frame_id (int): {
            'rh'               : np.float32(21, 4),  # [x_local, y_local, z_local, contact_ratio]
            'lh'               : np.float32(21, 4),  # [x_local, y_local, z_local, contact_ratio]
            'virtual_root_pos' : np.float32(3,),     # world-space virtual root origin
            'virtual_root_rot6d': np.float32(6,),    # world-space virtual root rotation (identity/world axes)
            'rh_wrist_pos_vr'  : np.float32(3,),     # rh wrist position relative to virtual root, in world axes
            'lh_wrist_pos_vr'  : np.float32(3,),     # lh wrist position relative to virtual root, in world axes
            'rh_wrist_rot6d_vr': np.float32(6,),     # world/virtual-root -> rh wrist-local frame rotation
            'lh_wrist_rot6d_vr': np.float32(6,),     # world/virtual-root -> lh wrist-local frame rotation
        }
    }

Usage:
    python scripts/extract_hand_joints.py                    # all sequences
    python scripts/extract_hand_joints.py --seq_file <path>
    python scripts/extract_hand_joints.py --workers 8
"""

import argparse
import glob
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, '/media/etri/02363C66363C5CBB/workspace/OakInk')
from manotorch.manolayer import ManoLayer
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_axis_angle

# ─── paths ────────────────────────────────────────────────────────────────────

MANO_ASSETS = "/media/etri/02363C66363C5CBB/workspace/OakInk/assets/mano_v1_2"
DATA_DIR    = "/media/etri/02363C66363C5CBB/workspace/OakInkv2"
MANO_VERTS  = 778
N_JOINTS    = 21
FINGERTIP_K = 20   # nearest verts used for fingertip contact ratio
J_WRIST = 0
J_THUMB_CMC = 1
J_INDEX_MCP = 5
J_MIDDLE_MCP = 9
J_PINKY_MCP = 17

# manotorch reorders joints to SNAP order:
#   SNAP idx → original MANO weight column (None = fingertip, no skinning col)
# reorder: [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
SNAP_TO_WEIGHT_COL = [0, 13, 14, 15, None, 1, 2, 3, None, 4, 5, 6, None, 10, 11, 12, None, 7, 8, 9, None]

# fingertip SNAP indices and their canonical vertex indices in MANO
TIP_SNAP_IDX   = [4, 8, 12, 16, 20]       # thumb, index, middle, ring, pinky
TIP_VERTS_R    = [745, 317, 444, 556, 673]
TIP_VERTS_L    = [745, 317, 445, 556, 673]
WORLD_R = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float32)  # rotate +90 deg about X so Z becomes the previous up axis


def apply_world_transform_pts(pts):
    arr = np.asarray(pts, dtype=np.float32)
    return arr @ WORLD_R.T


def apply_world_transform_rot(R):
    return WORLD_R @ np.asarray(R, dtype=np.float32)


def build_wrist_local_axes(joints_world, side):
    """
    Build a wrist-local orthonormal frame from world-space joints.

    Axis intent:
        x : back of hand -> palm
        y : pinky_MCP -> thumb_CMC
        z : wrist -> middle_MCP

    Returns:
        R : (3, 3) world -> local rotation for row-vector points,
            with columns [x_axis, y_axis, z_axis]
    """
    j = joints_world.astype(np.float64)
    wrist = j[J_WRIST]

    z_ref = j[J_MIDDLE_MCP] - wrist
    z_norm = np.linalg.norm(z_ref)
    if z_norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    z_axis = z_ref / z_norm

    y_ref = j[J_THUMB_CMC] - j[J_PINKY_MCP]
    y_axis = y_ref - np.dot(y_ref, z_axis) * z_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    y_axis /= y_norm

    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return np.eye(3, dtype=np.float32)
    x_axis /= x_norm

    # Keep the palm side on +X, matching the reference visualization.
    mean_x = np.mean((j[1:] - wrist) @ x_axis)
    if mean_x < 0.0:
        x_axis = -x_axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= (np.linalg.norm(y_axis) + 1e-8)

    # Ensure +Y always points from pinky toward thumb.
    if np.dot(y_axis, y_ref) < 0.0:
        y_axis = -y_axis
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-8)

    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def matrix_to_rot6d(R):
    # Store first two COLUMNS concatenated (column-major), so that
    # decoders treating r[:3] as col-0 and r[3:6] as col-1 are correct.
    # np default reshape is row-major → DO NOT use R[:, :2].reshape(6).
    R = np.asarray(R, dtype=np.float32)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def compute_virtual_root(rh_wrist_pos, lh_wrist_pos, R_rh_vr, R_lh_vr):
    """
    Build a wrist-pair virtual root in world coordinates.

    The virtual root is translation-only:
        origin : midpoint between left / right wrists
        axes   : identical to world axes

    With world-fixed virtual root axes, wrist positions are stored as world-axis
    offsets from the midpoint, and wrist rotations are stored as
    world/virtual-root -> wrist-local rotations (6D of R_world_to_local).

    R_rh_vr / R_lh_vr : (3,3) matrices whose COLUMNS are the wrist-local axes
                         expressed in world coords  (= R_local_to_world).
    R_world_to_local   : R_rh_vr.T  (row i is the i-th world axis projected
                         onto the local frame).
    6D representation  : first two columns of R_world_to_local, i.e.
                         (R_rh_vr.T)[:, :2].reshape(6).
    """
    rh = np.asarray(rh_wrist_pos, dtype=np.float64)
    lh = np.asarray(lh_wrist_pos, dtype=np.float64)
    root_pos = ((rh + lh) * 0.5).astype(np.float32)

    R_root = np.eye(3, dtype=np.float32)
    root_rot6d = matrix_to_rot6d(R_root)

    rh_pos_vr = (rh - root_pos.astype(np.float64)).astype(np.float32)
    lh_pos_vr = (lh - root_pos.astype(np.float64)).astype(np.float32)
    # R_world_to_local = R_local_to_world.T  →  6D of world→wrist-local rotation
    rh_rot_vr = matrix_to_rot6d(R_rh_vr.T)
    lh_rot_vr = matrix_to_rot6d(R_lh_vr.T)

    return {
        "virtual_root_pos": root_pos,
        "virtual_root_rot6d": root_rot6d,
        "rh_wrist_pos_vr": rh_pos_vr,
        "lh_wrist_pos_vr": lh_pos_vr,
        "rh_wrist_rot6d_vr": rh_rot_vr,
        "lh_wrist_rot6d_vr": lh_rot_vr,
    }


# ─── build static joint → vertex mapping ──────────────────────────────────────

def build_joint_vert_map(side):
    """
    Returns joint_vert_map: list of 21 np.int32 arrays, indexed in SNAP order.
    Non-tip joints : vertices whose dominant skinning weight belongs to that joint.
    Tip joints     : K nearest rest-pose vertices around the fingertip vertex.
    """
    path = os.path.join(MANO_ASSETS, "models",
                        "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl")
    with open(path, "rb") as f:
        m = pickle.load(f, encoding="latin1")

    weights    = m["weights"]     # (778, 16)  original MANO weight columns
    v_template = m["v_template"]  # (778, 3)

    # dominant weight column → vertex list
    col_to_verts = [[] for _ in range(16)]
    for vi in range(MANO_VERTS):
        col = int(weights[vi].argmax())
        col_to_verts[col].append(vi)

    # map SNAP joints to vertex lists
    joint_vert_map = [[] for _ in range(N_JOINTS)]
    for snap_j, w_col in enumerate(SNAP_TO_WEIGHT_COL):
        if w_col is not None:
            joint_vert_map[snap_j] = col_to_verts[w_col]

    # fingertip joints: K nearest vertices around the canonical tip vertex
    tip_verts = TIP_VERTS_R if side == "right" else TIP_VERTS_L
    tree = cKDTree(v_template)
    for i, snap_j in enumerate(TIP_SNAP_IDX):
        tip_v_pos = v_template[tip_verts[i]]
        _, nn_idx = tree.query(tip_v_pos, k=FINGERTIP_K)
        joint_vert_map[snap_j] = nn_idx.tolist()

    return [np.array(vids, dtype=np.int32) for vids in joint_vert_map]


# ─── MANO batch forward ───────────────────────────────────────────────────────

def quat_pose_to_aa(pose_coeffs):
    B = pose_coeffs.shape[0]
    return quaternion_to_axis_angle(pose_coeffs.reshape(-1, 4)).reshape(B, 48)


def batch_joints(raw_mano, frames, side):
    """
    Return {frame_id: {'joints': float32(21,3), 'wrist_pos': float32(3,), 'wrist_R_world': float32(3,3)}}.
    joints : world-space 21 joint positions
    """
    key_pose  = f"{side}__pose_coeffs"
    key_betas = f"{side}__betas"
    key_tsl   = f"{side}__tsl"

    q = torch.stack([raw_mano[fi][key_pose].squeeze(0)  for fi in frames])
    b = torch.stack([raw_mano[fi][key_betas].squeeze(0) for fi in frames])
    t = torch.stack([raw_mano[fi][key_tsl].squeeze(0)   for fi in frames])

    aa        = quat_pose_to_aa(q).float()
    side_name = "right" if side == "rh" else "left"
    mano      = ManoLayer(side=side_name, center_idx=0, mano_assets_root=MANO_ASSETS)
    with torch.no_grad():
        out = mano(aa, b.float())

    joints = (out.joints + t[:, None, :]).detach().numpy().astype(np.float32)  # (N,21,3)
    wrist_pos = t.detach().numpy().astype(np.float32)
    wrist_R_world = axis_angle_to_matrix(aa[:, :3]).detach().numpy().astype(np.float32)

    return {
        fi: {
            "joints": joints[i],
            "wrist_pos": wrist_pos[i],
            "wrist_R_world": wrist_R_world[i],
        }
        for i, fi in enumerate(frames)
    }


# ─── per-joint contact ratio ──────────────────────────────────────────────────

def joint_contact_ratios(contact_frame, side, valid_objs, joint_vert_map):
    """
    Returns np.float32(21,): per-joint contact ratio.
    contact ratio for joint j = (# contact verts in joint's vert set) / (# verts in set)
    """
    # union of contact hand vids across all objects for this side
    contact_vids = set()
    if contact_frame is not None:
        for oid in valid_objs:
            if oid in contact_frame.get(side, {}):
                entry = contact_frame[side][oid]
                if entry["is_contact"]:
                    contact_vids.update(entry["hand_vids"].tolist())

    ratios = np.zeros(N_JOINTS, dtype=np.float32)
    for j, vids in enumerate(joint_vert_map):
        if len(vids) == 0:
            continue
        n_contact = sum(1 for v in vids if v in contact_vids)
        ratios[j] = n_contact / len(vids)
    return ratios


# ─── main extraction ──────────────────────────────────────────────────────────

def extract_sequence(seq_file, data_dir, out_dir):
    torch.set_num_threads(2)

    seq_name = os.path.basename(seq_file).replace(".pkl", "")
    out_path  = os.path.join(out_dir, seq_name + ".pkl")


    t0 = time.time()

    with open(seq_file, "rb") as f:
        data = pickle.load(f)

    frames     = data["mocap_frame_id_list"]
    raw_mano   = data["raw_mano"]
    valid_objs = data["obj_list"]

    # contact annotation — required (run extract_contact.py first)
    contact_path = os.path.join(data_dir, "contact_anno", seq_name + ".pkl")
    if not os.path.exists(contact_path):
        print(f"  [skip] no contact_anno for {seq_name} — run extract_contact.py first")
        return None
    with open(contact_path, "rb") as f:
        contact_anno = pickle.load(f)

    # joint → vertex maps for rh and lh
    jvm_r = build_joint_vert_map("right")
    jvm_l = build_joint_vert_map("left")

    # batch MANO joint positions + wrist poses
    print(f"  [{seq_name}] joints rh ...", flush=True)
    rh_data = batch_joints(raw_mano, frames, "rh")
    print(f"  [{seq_name}] joints lh ...", flush=True)
    lh_data = batch_joints(raw_mano, frames, "lh")

    # per-frame assembly
    result = {}
    N = len(frames)
    for idx, fi in enumerate(frames):
        cf = contact_anno[fi]   # contact_anno is required; KeyError → explicit failure

        rh_j_world = apply_world_transform_pts(rh_data[fi]["joints"].astype(np.float32))  # (21,3)
        lh_j_world = apply_world_transform_pts(lh_data[fi]["joints"].astype(np.float32))

        rh_R = build_wrist_local_axes(rh_j_world, "rh")
        lh_R = build_wrist_local_axes(lh_j_world, "lh")
        rh_origin = apply_world_transform_pts(rh_data[fi]["wrist_pos"])
        lh_origin = apply_world_transform_pts(lh_data[fi]["wrist_pos"])
        rh_j_local = ((rh_j_world - rh_origin) @ rh_R).astype(np.float32)
        lh_j_local = ((lh_j_world - lh_origin) @ lh_R).astype(np.float32)
        lh_j_local[:, 0] *= -1.0
        
        rh_ratio = joint_contact_ratios(cf, "rh", valid_objs, jvm_r)  # (21,)
        lh_ratio = joint_contact_ratios(cf, "lh", valid_objs, jvm_l)

        # (21, 4): [x_local, y_local, z_local, contact_ratio]
        rh_4d = np.concatenate([rh_j_local, rh_ratio[:, None]], axis=1)
        lh_4d = np.concatenate([lh_j_local, lh_ratio[:, None]], axis=1)

        vr_data = compute_virtual_root(
            rh_origin,
            lh_origin,
            rh_R,
            lh_R,
        )

        result[fi] = {
            "rh": rh_4d,
            "lh": lh_4d,
            "virtual_root_pos": vr_data["virtual_root_pos"],
            "virtual_root_rot6d": vr_data["virtual_root_rot6d"],
            "rh_wrist_pos_vr": vr_data["rh_wrist_pos_vr"],
            "lh_wrist_pos_vr": vr_data["lh_wrist_pos_vr"],
            "rh_wrist_rot6d_vr": vr_data["rh_wrist_rot6d_vr"],
            "lh_wrist_rot6d_vr": vr_data["lh_wrist_rot6d_vr"],
        }

        if (idx + 1) % 1000 == 0 or (idx + 1) == N:
            print(f"  [{seq_name}] {idx+1}/{N}  ({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  [{seq_name}] saved → {out_path}  ({time.time()-t0:.1f}s)")
    return out_path


# ─── entry point ──────────────────────────────────────────────────────────────

def main(arg):
    data_dir = arg.data_dir
    out_dir  = os.path.join(data_dir, "hand_joint_anno")
    os.makedirs(out_dir, exist_ok=True)

    if arg.seq_file:
        seq_files = [arg.seq_file]
    else:
        seq_files = sorted(glob.glob(os.path.join(data_dir, "anno_preview", "*.pkl")))
        assert seq_files, f"No pkl files in {data_dir}/anno_preview"
        print(f"Found {len(seq_files)} sequences → {out_dir}")

    if arg.workers > 1 and len(seq_files) > 1:
        with ProcessPoolExecutor(max_workers=arg.workers) as ex:
            futs = {ex.submit(extract_sequence, sf, data_dir, out_dir): sf
                    for sf in seq_files}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print(f"ERROR {futs[fut]}: {e}")
    else:
        for sf in seq_files:
            extract_sequence(sf, data_dir, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, default=DATA_DIR)
    parser.add_argument("--seq_file",  type=str, default=None)
    parser.add_argument("--workers",   type=int, default=1)
    arg = parser.parse_args()
    main(arg)
