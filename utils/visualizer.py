import cv2
import numpy as np

from .geometry import (
    params_8_to_params_7,
    params_10_to_params_7,
    params_7_to_corners_8,
)


def project_3d_to_2d(corners_3d, K):
    """
    Project 3D points to image plane.

    Args:
        corners_3d: (N, 3)
        K: camera intrinsic matrix (3, 3)

    Returns:
        pts_2d: (N, 2)
    """
    corners_3d = np.asarray(corners_3d, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)

    if corners_3d.ndim != 2 or corners_3d.shape[1] != 3:
        raise ValueError(f"Expected corners_3d shape (N, 3), got {corners_3d.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"Expected K shape (3, 3), got {K.shape}")

    z = corners_3d[:, 2:3].copy()
    z[z < 0.1] = 0.1

    pts_h = (K @ corners_3d.T).T  # (N, 3)
    pts_2d = pts_h[:, :2] / np.clip(pts_h[:, 2:3], 1e-6, None)
    return pts_2d.astype(np.float32)


def draw_projected_box(image, params_7, K, color=(0, 255, 0), thickness=2):
    """
    Draw a projected 3D wireframe box on an image.

    Args:
        image: uint8 RGB or BGR image
        params_7: [x, y, z, w, l, h, yaw]
        K: camera intrinsics
    """
    img = image.copy()
    corners_3d = params_7_to_corners_8(params_7)
    pts_2d = project_3d_to_2d(corners_3d, K).astype(np.int32)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    h, w = img.shape[:2]

    def clip_pt(pt):
        x = int(np.clip(pt[0], 0, w - 1))
        y = int(np.clip(pt[1], 0, h - 1))
        return x, y

    for start, end in edges:
        p1 = clip_pt(pts_2d[start])
        p2 = clip_pt(pts_2d[end])
        cv2.line(img, p1, p2, color, thickness)

    return img


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def _simple_grid_nms(sorted_indices, min_dist=1.5):
    """
    Simple grid-space suppression.
    """
    kept = []
    for y, x in sorted_indices:
        if any(np.linalg.norm(np.array([y, x]) - np.array(prev)) < min_dist for prev in kept):
            continue
        kept.append((y, x))
    return kept


def visualize_multi_objects(
    rgb_img,
    pred_map,
    K,
    threshold=0.5,
    grid_size=(15, 20),
    range_x=(-10.0, 10.0),
    range_z=(0.0, 20.0),
    use_offsets=True,
    nms_dist=1.5,
):
    """
    Visualize multi-object detections from model output.

    Args:
        rgb_img: [H, W, 3], float in [0, 1] or uint8
        pred_map:
            either [11, Gh, Gw] -> [heatmap, 10 regression channels]
            or     [9,  Gh, Gw] -> [heatmap, 8 regression channels]
        K: camera intrinsics
        threshold: confidence threshold on heatmap
        grid_size: detector output grid size
        range_x, range_z: world coordinate ranges used in encoding/decoding
        use_offsets: if True and pred_map has 11 channels, decode with offsets
        nms_dist: suppression distance in grid cells

    Returns:
        vis_img_rgb: RGB uint8 image with projected boxes
    """
    pred_map = np.asarray(pred_map, dtype=np.float32)

    if rgb_img.dtype != np.uint8:
        vis_img = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)
    else:
        vis_img = rgb_img.copy()

    # OpenCV drawing in BGR
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    heatmap = pred_map[0]
    if heatmap.min() < 0.0 or heatmap.max() > 1.0:
        heatmap = _sigmoid_np(heatmap)

    indices = np.argwhere(heatmap > threshold)
    indices = sorted(indices, key=lambda idx: heatmap[idx[0], idx[1]], reverse=True)
    indices = _simple_grid_nms(indices, min_dist=nms_dist)

    for y, x in indices:
        if pred_map.shape[0] >= 11 and use_offsets:
            p10 = pred_map[1:11, y, x]
            params_7 = params_10_to_params_7(
                p10,
                grid_pos=(int(y), int(x)),
                grid_size=grid_size,
                range_x=range_x,
                range_z=range_z,
            )
        elif pred_map.shape[0] >= 9:
            p8 = pred_map[1:9, y, x]
            params_7 = params_8_to_params_7(p8)
        else:
            raise ValueError(
                f"pred_map must have at least 9 channels, got shape {pred_map.shape}"
            )

        vis_img = draw_projected_box(vis_img, params_7, K, color=(0, 255, 0), thickness=2)

    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)