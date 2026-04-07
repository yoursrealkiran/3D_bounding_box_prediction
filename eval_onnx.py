import os
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

from configs.load_config import load_config
from utils.geometry import params_10_to_params_7
from utils.visualizer import draw_projected_box


def load_onnx_session(onnx_path, use_gpu=True):
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)

    print("ONNX providers:", session.get_providers())
    print("ONNX inputs :", [x.name for x in session.get_inputs()])
    print("ONNX outputs:", [x.name for x in session.get_outputs()])

    return session


def preprocess_scene(rgb_raw, pc_raw, target_size=(480, 640), z_shift=1.5):
    """
    Must match dataset preprocessing for evaluation consistency.
    Returns NumPy arrays ready for ONNX: [1, C, H, W], float32
    """
    rgb_np = rgb_raw.astype(np.float32) / 255.0
    pc_np = pc_raw.astype(np.float32).copy()

    # Same alignment as dataset.py
    pc_np[0, :, :] = -pc_np[0, :, :]
    pc_np[1, :, :] = -pc_np[1, :, :]
    pc_np[2, :, :] -= z_shift

    rgb_t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    pc_t = torch.from_numpy(pc_np).unsqueeze(0)                      # [1,3,H,W]

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

    return rgb_t.numpy().astype(np.float32), pc_t.numpy().astype(np.float32)


def extract_local_peaks(prob_map, score_threshold=0.15, pool_kernel=3):
    """
    Local maxima extraction using torch max pooling on CPU.
    prob_map: numpy array [H, W]
    """
    prob_t = torch.from_numpy(prob_map).float().unsqueeze(0).unsqueeze(0)
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


def topk_peaks(peaks, top_k=20):
    peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
    return peaks[:top_k]


def simple_grid_nms(detections, min_dist=2.0):
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


def is_valid_box(params_7, range_x, range_z, z_shift=1.5, min_size=0.01, max_size=3.0):
    x, y, z, w, l, h, yaw = params_7

    if not np.isfinite(params_7).all():
        return False

    if not (range_x[0] <= x <= range_x[1]):
        return False

    z_min_world = range_z[0] + z_shift
    z_max_world = range_z[1] + z_shift
    if not (z_min_world <= z <= z_max_world):
        return False

    for dim in [w, l, h]:
        if dim < min_size or dim > max_size:
            return False

    return True


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_evaluation_onnx(onnx_path, scene_dir):
    cfg = load_config("configs/config.yaml")

    session = load_onnx_session(
        onnx_path,
        use_gpu=bool(cfg["eval"].get("use_onnx_gpu", True)),
    )

    target_size = tuple(cfg["data"]["target_size"])
    grid_size = tuple(cfg["data"]["grid_size"])
    range_x = tuple(cfg["data"]["range_x"])
    range_z = tuple(cfg["data"]["range_z"])
    z_shift = float(cfg["data"].get("z_shift", 1.5))

    score_threshold = float(cfg["eval"].get("score_threshold", 0.15))
    seg_threshold = float(cfg["eval"].get("seg_threshold", 0.10))
    nms_grid_distance = float(cfg["eval"].get("nms_grid_distance", 2.0))
    top_k = int(cfg["eval"].get("top_k", 20))
    peak_pool_kernel = int(cfg["eval"].get("peak_pool_kernel", 3))
    save_dir = cfg["eval"].get("save_dir", "./results_onnx")
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

    print(f"Starting ONNX evaluation on {len(scenes)} scenes...")

    for scene in scenes:
        scene_path = os.path.join(scene_dir, scene)
        rgb_raw = np.array(Image.open(os.path.join(scene_path, "rgb.jpg")).convert("RGB"))
        pc_raw = np.load(os.path.join(scene_path, "pc.npy")).astype(np.float32)

        rgb_np, pc_np = preprocess_scene(
            rgb_raw=rgb_raw,
            pc_raw=pc_raw,
            target_size=target_size,
            z_shift=z_shift,
        )

        # ONNX inference
        cls_output, box_output, seg_output = session.run(
            ["cls_output", "box_output", "seg_output"],
            {
                "rgb_input": rgb_np,
                "pc_input": pc_np,
            },
        )

        probs = sigmoid_np(cls_output[0, 0])     # [Gh, Gw]
        pred_maps = box_output[0]                # [10, Gh, Gw]
        seg_probs = sigmoid_np(seg_output[0, 0]) # [Gh, Gw]

        print(
            f"[{scene}] heatmap max={probs.max():.3f}, mean={probs.mean():.3f}, "
            f"seg max={seg_probs.max():.3f}, mean={seg_probs.mean():.3f}"
        )

        peaks = extract_local_peaks(
            probs,
            score_threshold=score_threshold,
            pool_kernel=peak_pool_kernel,
        )
        peaks = topk_peaks(peaks, top_k=top_k)

        filtered_by_seg = 0
        filtered_by_box = 0
        detections = []

        for y, x, score in peaks:
            if seg_probs[y, x] < seg_threshold:
                filtered_by_seg += 1
                continue

            p10 = pred_maps[:, y, x]

            p7 = params_10_to_params_7(
                p10,
                grid_pos=(int(y), int(x)),
                grid_size=grid_size,
                range_x=range_x,
                range_z=range_z,
            )

            p7[2] += z_shift

            if not is_valid_box(
                p7,
                range_x=range_x,
                range_z=range_z,
                z_shift=z_shift,
                min_size=0.01,
                max_size=3.0,
            ):
                filtered_by_box += 1
                continue

            detections.append(
                {
                    "score": float(score),
                    "seg_score": float(seg_probs[y, x]),
                    "params": p7,
                    "grid_pos": (int(y), int(x)),
                }
            )

        final_detections = simple_grid_nms(detections, min_dist=nms_grid_distance)

        print(
            f"[{scene}] peaks={len(peaks)}, seg_filtered={filtered_by_seg}, "
            f"box_filtered={filtered_by_box}, valid={len(detections)}, kept={len(final_detections)}"
        )

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

    print(f"Saved ONNX visualizations to: {save_dir}")


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")

    onnx_path = "checkpoints/fusion_bbox_model.onnx"
    data_path = cfg["data"]["root_dir"]

    run_evaluation_onnx(
        onnx_path=onnx_path,
        scene_dir=data_path,
    )