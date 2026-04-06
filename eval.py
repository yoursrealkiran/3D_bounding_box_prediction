import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.load_config import load_config
from models.pipeline_main import Fusion3DDetector
from utils.geometry import params_10_to_params_7
from utils.visualizer import draw_projected_box


def load_model(checkpoint_path, cfg, device):
    model = Fusion3DDetector(
        freeze_rgb_layers=bool(cfg["model"].get("freeze_rgb_layers", True)),
        head_dropout=float(cfg["model"].get("dropout", 0.2)),
        output_grid_size=tuple(cfg["data"]["grid_size"]),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k[len("model."):]] = v
        else:
            cleaned_state_dict[k] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model


def preprocess_scene(rgb_raw, pc_raw, target_size=(480, 640), z_shift=1.5):
    """
    Must match dataset preprocessing for evaluation consistency.
    """
    rgb_np = rgb_raw.astype(np.float32) / 255.0
    pc_np = pc_raw.astype(np.float32).copy()

    # Same alignment as dataset.py
    pc_np[0, :, :] = -pc_np[0, :, :]
    pc_np[1, :, :] = -pc_np[1, :, :]
    pc_np[2, :, :] -= z_shift

    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)
    pc_t = torch.from_numpy(pc_np).unsqueeze(0)

    rgb_t = F.interpolate(
        rgb_t,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    pc_t = F.interpolate(
        pc_t,
        size=target_size,
        mode="nearest",
    )

    return rgb_t, pc_t


def extract_local_peaks(prob_map, score_threshold=0.35, pool_kernel=3):
    """
    Local maxima extraction using max pooling.

    Args:
        prob_map: numpy array [H, W]
        score_threshold: minimum probability to keep
        pool_kernel: local neighborhood size

    Returns:
        list of tuples (y, x, score)
    """
    prob_t = torch.from_numpy(prob_map).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    pooled = F.max_pool2d(
        prob_t,
        kernel_size=pool_kernel,
        stride=1,
        padding=pool_kernel // 2,
    )

    peak_mask = (prob_t == pooled) & (prob_t >= score_threshold)
    peak_mask = peak_mask.squeeze(0).squeeze(0).cpu().numpy()

    ys, xs = np.where(peak_mask)
    peaks = [(int(y), int(x), float(prob_map[y, x])) for y, x in zip(ys, xs)]
    return peaks


def topk_peaks(peaks, top_k=10):
    """
    Keep only the top-K peaks by score.
    """
    peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
    return peaks[:top_k]


def simple_grid_nms(detections, min_dist=2.0):
    """
    Grid-space NMS.

    detections: list of dicts with:
        - score
        - grid_pos
        - params
    """
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in detections:
        curr = np.array(det["grid_pos"], dtype=np.float32)
        suppress = False

        for prev in kept:
            prev_grid = np.array(prev["grid_pos"], dtype=np.float32)
            if np.linalg.norm(curr - prev_grid) < min_dist:
                suppress = True
                break

        if not suppress:
            kept.append(det)

    return kept


def is_valid_box(params_7, range_x, range_z, z_shift=1.5, min_size=0.01, max_size=1.5):
    """
    Simple sanity filter for decoded boxes.
    params_7 = [x, y, z, w, l, h, yaw]
    """
    x, y, z, w, l, h, yaw = params_7

    if not np.isfinite(params_7).all():
        return False

    if not (range_x[0] <= x <= range_x[1]):
        return False

    # z here is after undoing z-shift, so compare with original-space adjusted range
    z_min_world = range_z[0] + z_shift
    z_max_world = range_z[1] + z_shift
    if not (z_min_world <= z <= z_max_world):
        return False

    for dim in [w, l, h]:
        if dim < min_size or dim > max_size:
            return False

    return True


def run_evaluation(checkpoint_path, scene_dir):
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, cfg, device)

    target_size = tuple(cfg["data"]["target_size"])
    grid_size = tuple(cfg["data"]["grid_size"])
    range_x = tuple(cfg["data"]["range_x"])
    range_z = tuple(cfg["data"]["range_z"])
    z_shift = float(cfg["data"].get("z_shift", 1.5))

    score_threshold = float(cfg["eval"].get("score_threshold", 0.35))
    seg_threshold = float(cfg["eval"].get("seg_threshold", 0.3))
    nms_grid_distance = float(cfg["eval"].get("nms_grid_distance", 2.5))
    top_k = int(cfg["eval"].get("top_k", 8))
    peak_pool_kernel = int(cfg["eval"].get("peak_pool_kernel", 3))
    save_dir = cfg["eval"].get("save_dir", "./results")
    os.makedirs(save_dir, exist_ok=True)

    cam_cfg = cfg["eval"]["camera_intrinsics"]
    K = np.array(
        [
            [float(cam_cfg["fx"]), 0.0, float(cam_cfg["cx"])],
            [0.0, float(cam_cfg["fy"]), float(cam_cfg["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    scenes = sorted(
        [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]
    )

    print(f"Starting evaluation on {len(scenes)} scenes...")

    for scene in scenes:
        scene_path = os.path.join(scene_dir, scene)
        rgb_raw = np.array(Image.open(os.path.join(scene_path, "rgb.jpg")).convert("RGB"))
        pc_raw = np.load(os.path.join(scene_path, "pc.npy")).astype(np.float32)

        rgb_t, pc_t = preprocess_scene(
            rgb_raw=rgb_raw,
            pc_raw=pc_raw,
            target_size=target_size,
            z_shift=z_shift,
        )

        rgb_t = rgb_t.to(device)
        pc_t = pc_t.to(device)

        with torch.no_grad():
            logits, bboxes, seg = model(rgb_t, pc_t)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()      # [Gh, Gw]
            pred_maps = bboxes.squeeze(0).cpu().numpy()                            # [10, Gh, Gw]
            seg_probs = torch.sigmoid(seg).squeeze(0).squeeze(0).cpu().numpy()     # [Gh, Gw]

        # 1. Peak extraction with max-pooling
        peaks = extract_local_peaks(
            probs,
            score_threshold=score_threshold,
            pool_kernel=peak_pool_kernel,
        )

        # 2. Keep only strongest peaks
        peaks = topk_peaks(peaks, top_k=top_k)

        detections = []
        for y, x, score in peaks:
            
            if seg_probs[y, x] < seg_threshold:
                continue

            p10 = pred_maps[:, y, x]

            p7 = params_10_to_params_7(
                p10,
                grid_pos=(int(y), int(x)),
                grid_size=grid_size,
                range_x=range_x,
                range_z=range_z,
            )

            # Undo z-shift for visualization in original coordinates
            p7[2] += z_shift

            if not is_valid_box(
                p7,
                range_x=range_x,
                range_z=range_z,
                z_shift=z_shift,
                min_size=0.01,
                max_size=1.5,
            ):
                continue

            detections.append(
                {
                    "score": float(score),
                    "seg_score": float(seg_probs[y, x]),
                    "params": p7,
                    "grid_pos": (int(y), int(x)),
                }
            )

        # 3. Grid-space NMS
        final_detections = simple_grid_nms(detections, min_dist=nms_grid_distance)

        if final_detections:
            best_det = final_detections[0]
            print(
                f"[{scene}] raw_peaks={len(peaks)}, valid={len(detections)}, kept={len(final_detections)}, "
                f"best_score={best_det['score']:.3f}, seg={best_det['seg_score']:.3f}, "
                f"grid={best_det['grid_pos']}, "
                f"center=({best_det['params'][0]:.2f}, {best_det['params'][1]:.2f}, {best_det['params'][2]:.2f})"
            )
        else:
            print(f"[{scene}] no detections after peak extraction/NMS")

        vis_img = rgb_raw.copy()
        for det in final_detections:
            vis_img = draw_projected_box(
                vis_img,
                det["params"],
                K,
                color=(0, 255, 0),
                thickness=2,
            )

        save_path = os.path.join(save_dir, f"{scene}_3d_detect.png")
        Image.fromarray(vis_img).save(save_path)

    print(f"Saved visualizations to: {save_dir}")


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")

    ckpt = os.path.join(cfg["logging"]["save_dir"], "last.ckpt")
    data_path = cfg["data"]["root_dir"]

    run_evaluation(
        checkpoint_path=ckpt,
        scene_dir=data_path,
    )