import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
import cv2
from moviepy import ImageSequenceClip
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.global_hydra import GlobalHydra
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.global_hydra import GlobalHydra

from tactile_ssl.data.brainco_tactile import build_combined_chain, compute_combined_fk, FINGERTIP_LINKS
from tactile_ssl.downstream_task.brainco_grasp_sl import BraincoGraspDetectionSLModule


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Grasp Detection on an episode")
    parser.add_argument("--target", type=str, required=True, help="Path to the target episode folder")
    # For Hydra style arguments (+experiment=...)
    parser.add_argument('hydra_args', nargs=argparse.REMAINDER)
    return parser.parse_args()



def load_episode_data(ep_path: Path, config: DictConfig):
    """
    Load an episode data identically to BraincoGraspDetectionDataset._load_episode, 
    but return without windowing so we can run sliding inference on it.
    """
    data_json_path = ep_path / "data.json"
    with open(data_json_path, "r") as f:
        raw_data = json.load(f)

    frames = raw_data["data"]
    num_frames = len(frames)

    # Reconstruct kinematics chain
    brainco_urdf_path = Path("dataset/brainco/urdf")
    chain = build_combined_chain(brainco_urdf_path)

    # Extract joint positions
    left_arm_list, right_arm_list = [], []
    left_ee_list, right_ee_list = [], []
    for frame in frames:
        states = frame["states"]
        left_arm_list.append(states["left_arm"]["qpos"])
        right_arm_list.append(states["right_arm"]["qpos"])
        left_ee_list.append(states["left_ee"]["qpos"])
        right_ee_list.append(states["right_ee"]["qpos"])

    left_arm_array = np.array(left_arm_list, dtype=np.float32)
    right_arm_array = np.array(right_arm_list, dtype=np.float32)
    left_ee_array = np.array(left_ee_list, dtype=np.float32)
    right_ee_array = np.array(right_ee_list, dtype=np.float32)

    # FK for fingertip positions
    fingertip_positions_list = []
    wrist_positions_list = []
    for i in range(num_frames):
        fk_res = compute_combined_fk(
            chain,
            left_arm_array[i], right_arm_array[i],
            left_ee_array[i], right_ee_array[i],
        )
        # Wrists
        left_wrist = fk_res["left_rubber_hand"].get_matrix()[:, :3, 3].squeeze(0).numpy()
        right_wrist = fk_res["right_rubber_hand"].get_matrix()[:, :3, 3].squeeze(0).numpy()
        wrist_positions_list.append(np.stack([left_wrist, right_wrist], axis=0))

        # Fingertips
        frame_positions = []
        for link_suffix in FINGERTIP_LINKS:
            frame_positions.append(fk_res[f"left_{link_suffix}"].get_matrix()[:, :3, 3].squeeze(0).numpy())
        for link_suffix in FINGERTIP_LINKS:
            frame_positions.append(fk_res[f"right_{link_suffix}"].get_matrix()[:, :3, 3].squeeze(0).numpy())
        fingertip_positions_list.append(np.stack(frame_positions, axis=0))

    fp_world = np.array(fingertip_positions_list, dtype=np.float32)  # (N, 10, 3)
    wrist_positions = np.array(wrist_positions_list, dtype=np.float32)  # (N, 2, 3)

    # 6D fingertip positions (relative to own + opposite wrist)
    left_idxs = [0, 1, 2, 3, 4]
    right_idxs = [5, 6, 7, 8, 9]
    fingertip_6d = np.zeros((num_frames, 10, 6), dtype=np.float32)
    for idx in left_idxs:
        fingertip_6d[:, idx, :3] = fp_world[:, idx, :] - wrist_positions[:, 0, :]
        fingertip_6d[:, idx, 3:] = fp_world[:, idx, :] - wrist_positions[:, 1, :]
    for idx in right_idxs:
        fingertip_6d[:, idx, :3] = fp_world[:, idx, :] - wrist_positions[:, 1, :]
        fingertip_6d[:, idx, 3:] = fp_world[:, idx, :] - wrist_positions[:, 0, :]

    # Load tactile data
    tactile_list = []
    for frame in frames:
        tactile_info = frame["tactiles"]
        if isinstance(tactile_info["left_ee"], str):
            left_path = ep_path / tactile_info["left_ee"]
            left_tactile = np.load(str(left_path)).reshape(-1, 4)
        else:
            left_tactile = np.array(tactile_info["left_ee"]).reshape(-1, 4)

        if isinstance(tactile_info["right_ee"], str):
            right_path = ep_path / tactile_info["right_ee"]
            right_tactile = np.load(str(right_path)).reshape(-1, 4)
        else:
            right_tactile = np.array(tactile_info["right_ee"]).reshape(-1, 4)

        tactile = np.concatenate([left_tactile, right_tactile], axis=0)
        tactile_list.append(tactile)

    tactile_array = np.array(tactile_list, dtype=np.float32)  # (N, num_sensors, 4)

    # Replace invalid values
    invalid_mask = tactile_array[..., 2] == 65535
    tactile_array[..., 2][invalid_mask] = -1

    return tactile_array, fingertip_6d, num_frames

# Let's replace the whole main block and bottom part
target_path_global = None

def parse_args_custom():
    """
    Manually parse --target first so Hydra doesn't complain,
    and then remove it from sys.argv so Hydra can parse the rest.
    Also inject Hydra overrides to prevent creating a new experiment output directory.
    """
    import argparse
    import sys
    from pathlib import Path
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target", type=str, required=True, help="Path to episode folder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Explicit path to the checkpoint file to load")
    
    args, unknown = parser.parse_known_args()
    
    # Keep only the script name and unknown args for Hydra
    # Add Hydra overrides to disable output directory creation
    sys.argv = [sys.argv[0]] + unknown + ["hydra.run.dir=.", "hydra.output_subdir=null"]
    
    global target_path_global
    target_path_global = Path(args.target)
    
    global checkpoint_path_global
    checkpoint_path_global = Path(args.checkpoint)
    
    return args


@hydra.main(version_base="1.3", config_path="config", config_name="default_task.yaml")
def main(cfg: DictConfig):
    global target_path_global
    if target_path_global is None:
        raise ValueError("Must provide --target <path>")
    target_path = Path(target_path_global)
    
    if not target_path.exists():
        raise FileNotFoundError(f"Target folder {target_path} does not exist")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Set checkpoint from arguments
    OmegaConf.set_struct(cfg, False)
    global checkpoint_path_global
    
    if not checkpoint_path_global.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path_global}")
        
    print(f"Using explicitly provided checkpoint: {checkpoint_path_global}")
    
    OmegaConf.set_struct(cfg, True)
    
    print("Instantiating model...")
    model = hydra.utils.instantiate(cfg.task)
    model.to(device) # Move model to device after instantiation
    model.eval()

    # Load checkpoint
    print(f"Loading task checkpoint from {checkpoint_path_global} ...")
    ckpt = torch.load(checkpoint_path_global, map_location=device, weights_only=False)
    print(ckpt['model'].keys())
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    print(f"Loading data from {target_path} ...")
    tactile_array, fingertip_6d, num_frames = load_episode_data(target_path, cfg)
    
    dataset_cfg = cfg.data
    window_time = dataset_cfg.window_time
    interpolating_freq = dataset_cfg.interpolating_freq
    num_frames_per_window = int(round(window_time * interpolating_freq))
    
    shift_per_window = 1 
    
    frame_predictions = np.zeros(num_frames, dtype=int) - 1 
    
    print(f"Running inference on {num_frames} frames...")
    
    batch_size = 64
    windows_sensor = []
    windows_poses = []
    window_indices = []
    
    with torch.no_grad():
        for start in range(0, num_frames - num_frames_per_window + 1, shift_per_window):
            end = start + num_frames_per_window
            sensor_win = tactile_array[start:end]
            pos_win = fingertip_6d[start:end]
            
            windows_sensor.append(sensor_win)
            windows_poses.append(pos_win)
            window_indices.append(start)
            
            if len(windows_sensor) == batch_size or start == num_frames - num_frames_per_window:
                batch_sensor = torch.from_numpy(np.array(windows_sensor)).float().to(device) 
                batch_poses = torch.from_numpy(np.array(windows_poses)).float().to(device)   
                
                batch = {
                    "sensor": batch_sensor,
                    "sensor_poses": batch_poses
                }
                
                logits = model(batch) 
                preds = logits.argmax(dim=-1).cpu().numpy()
                
                for i, p in enumerate(preds):
                    idx = window_indices[i]
                    end_idx = idx + num_frames_per_window - 1
                    if end_idx < num_frames:
                        frame_predictions[end_idx] = p
                
                windows_sensor = []
                windows_poses = []
                window_indices = []

    first_pred_idx = np.where(frame_predictions != -1)[0]
    if len(first_pred_idx) > 0:
        first_idx = first_pred_idx[0]
        first_val = frame_predictions[first_idx]
        frame_predictions[:first_idx] = first_val
        
        current_pred = first_val
        for i in range(first_idx, num_frames):
            if frame_predictions[i] == -1:
                frame_predictions[i] = current_pred
            else:
                current_pred = frame_predictions[i]
    else:
        print("No predictions made. Using default 0.")
        frame_predictions[:] = 0

    colors_dir = target_path / "colors"
    if not colors_dir.exists():
        raise FileNotFoundError(f"Colors directory {colors_dir} not found.")
        
    img_files = sorted([f for f in colors_dir.iterdir() if f.is_file() and f.suffix in ['.jpg', '.png']])
    if len(img_files) == 0:
        raise ValueError(f"No images found in {colors_dir}")
        
    print(f"Found {len(img_files)} images, mapping to {num_frames} frames.")
    
    min_len = min(len(img_files), num_frames)
    
    # Let's define utility to generate hand plot
    # fingertip_6d shape (num_frames, 10, 6). 0->4 left, 5->9 right.
    # tactile_array shape (num_frames, 10, 4) since num_sensors is 10 and each has 4 nodes. Wait. 
    # Let's check tactile_array shape in load_episode_data:
    # tactile_list.append(tactile); tactile_array = np.array(tactile_list)
    # tactile is (num_sensors, 4) which is 10x4.
    
    def render_hand_plot(sensors, is_left=True):
        # sensors: (5, 4)
        
        # We will plot a fixed 2D layout for the fingertips
        fig, ax = plt.subplots(figsize=(2, 3), dpi=100)
        
        # Extract normal force (first index) per finger
        sensor_vals = sensors[:, 0] # (5,)
        
        # Extract proximity sensor (fourth index) per finger
        prox_vals = sensors[:, 3] # (5,)
        
        # Map normal force to color and size
        # Assuming sensor values range roughly -1 to 5 but typically 0-1 for normalized
        sizes = 50 + np.clip(sensor_vals, 0, 10) * 100
        colors = plt.cm.Reds(np.clip(sensor_vals, 0, 5) / 5.0)
        
        # Map proximity to color and size
        prox_sizes = 50 + np.clip(prox_vals, 0, 10) * 100
        prox_colors = plt.cm.Blues(np.clip(prox_vals, 0, 5) / 5.0)
        
        # Fixed coordinates for [Thumb, Index, Middle, Ring, Pinky]
        if is_left:
            # User requested inverted x-axis for left hand.
            x = np.array([-0.08, -0.04, 0.0, 0.04, 0.08])
        else:
            x = np.array([-0.08, -0.04, 0.0, 0.04, 0.08])
            
        y = np.array([-0.02, 0.10, 0.12, 0.10, 0.06])
        
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors='red')
        ax.scatter(x, y, s=prox_sizes, c=prox_colors, alpha=0.3, edgecolors='blue')
        
        # Draw small markers for labels and sensor values
        finger_names = ['T', 'I', 'M', 'R', 'P'] # Thumb, Index, Middle, Ring, Pinky
        for fn, fx, fy, nv, pv in zip(finger_names, x, y, sensor_vals, prox_vals):
            ax.text(fx, fy, fn, fontsize=8, ha='center', va='center')
            # Display both normal and proximity values just below the finger
            text_str = f"N:{nv:.1f}\nP:{pv:.1f}"
            ax.text(fx, fy - 0.07, text_str, fontsize=6, ha='center', va='top', color='black')
        
        ax.set_title("Left Hand" if is_left else "Right Hand")
        
        # Set roughly fixed limits so it doesn't jump
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(-0.15, 0.2)
        ax.axis('off')
        
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_arr = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img_arr = img_arr.reshape(canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        
        return img_arr[:, :, :3] # Keep RGB

    rendered_frames = []
    
    print("Rendering video frames...")
    for i in range(min_len):
        img_path = str(img_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Render left/right hand plots for frame i
        l_plot = render_hand_plot(tactile_array[i, :5], is_left=True)
        r_plot = render_hand_plot(tactile_array[i, 5:], is_left=False)
        
        # Resize plots to match image height if needed
        # Assume img shape e.g., 480x640x3. l_plot shape e.g., 300x200x3
        h_img, w_img, _ = img.shape
        scale_l = h_img / l_plot.shape[0]
        dim_l = (int(l_plot.shape[1] * scale_l), h_img)
        l_plot_img = cv2.resize(l_plot, dim_l, interpolation=cv2.INTER_AREA)
        
        scale_r = h_img / r_plot.shape[0]
        dim_r = (int(r_plot.shape[1] * scale_r), h_img)
        r_plot_img = cv2.resize(r_plot, dim_r, interpolation=cv2.INTER_AREA)
        
        # Concatenate horizontally: left_plot | img | right_plot
        combined_img = np.concatenate([l_plot_img, img, r_plot_img], axis=1)
        
        pred = frame_predictions[i]
        
        color = (0, 255, 0) if pred == 1 else (255, 0, 0)
        thickness = 10
        cv2.rectangle(combined_img, (0, 0), (combined_img.shape[1]-1, combined_img.shape[0]-1), color, thickness)
        
        text = "Success" if pred == 1 else "Fail"
        # Draw on the central image part explicitly
        cv2.putText(combined_img, f"Pred: {text}", (l_plot_img.shape[1] + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        rendered_frames.append(combined_img)
        
    print(f"Saving video to output for {target_path.name}...")
    clip = ImageSequenceClip(rendered_frames, fps=30) 
    out_path = str(Path(f"inference_video_{target_path.name}_v2.mp4"))
    clip.write_videofile(out_path, logger=None)
    
    print(f"Done! Video saved to {out_path}")

if __name__ == "__main__":
    from tactile_ssl.data.d360.utils import get_weights, get_experiment_name, get_modality_tag, get_modality_used_tag
    if not OmegaConf.has_resolver("int_multiply"):
        OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))
    if not OmegaConf.has_resolver("int_divide"):
        OmegaConf.register_new_resolver("int_divide", lambda a, b: a // b)
    if not OmegaConf.has_resolver("d360_expt_name"):
        OmegaConf.register_new_resolver("d360_expt_name", get_experiment_name)
    if not OmegaConf.has_resolver("d360_modal_tag"):
        OmegaConf.register_new_resolver("d360_modal_tag", get_modality_tag)
    if not OmegaConf.has_resolver("d360_modal_used_tag"):
        OmegaConf.register_new_resolver("d360_modal_used_tag", get_modality_used_tag)
    if not OmegaConf.has_resolver("capitalize"):
        OmegaConf.register_new_resolver("capitalize", lambda s: s.title())
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
        
    parse_args_custom()
    main()
