import numpy as np
import torch

def draw_msra_gaussian(heatmap, center, sigma):
    """
    Draws a 2D Gaussian onto a heatmap.
    center: (grid_x, grid_y)
    sigma: radius of the object in grid cells
    """
    tmp_size = int(3 * sigma)
    mu_x, mu_y = int(center[0]), int(center[1])
    h, w = heatmap.shape

    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)

    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    )
    return heatmap

def corners_8_to_params_8(corners):
    """Converts 8 corners [8, 3] to [x, y, z, w, l, h, sin, cos]"""
    center = np.mean(corners, axis=0)
    
    # Using your specific indices for dimensions
    w = np.linalg.norm(corners[0] - corners[4])
    l = np.linalg.norm(corners[0] - corners[3])
    h = np.linalg.norm(corners[0] - corners[1])
    
    direction_vector = corners[3] - corners[0]
    yaw = np.arctan2(direction_vector[2], direction_vector[0])
    
    return np.array([center[0], center[1], center[2], w, l, h, np.sin(yaw), np.cos(yaw)], dtype=np.float32)

def params_10_to_params_7(p10, grid_pos, grid_size=(15, 20)):
    """
    Converts model output (8 params + 2 offsets) to 7D world params.
    p10: [x, y, z, w, l, h, sin, cos, off_x, off_z]
    grid_pos: (grid_y, grid_x) indices where this was predicted
    """
    # 1. Recover Yaw
    yaw = np.arctan2(p10[6], p10[7])
    
    # 2. Refine Center X and Z using Predicted Offsets
    # This reverses the logic: norm_x = (center_x + 0.5) * grid_size_w
    # center_x = (grid_x + off_x) / grid_size_w - 0.5
    gy, gx = grid_pos
    off_x, off_z = p10[8], p10[9]
    
    refined_x = (gx + off_x) / grid_size[1] - 0.5
    refined_z = (gy + off_z) / grid_size[0] - 0.5
    
    # Note: center[1] is Y (height), which doesn't use grid offsets in this setup
    return np.array([refined_x, p10[1], refined_z, p10[3], p10[4], p10[5], yaw], dtype=np.float32)

def params_8_to_params_7(p8):
    """Fallback for when you don't have offsets (e.g. simple validation)"""
    yaw = np.arctan2(p8[6], p8[7])
    return np.array([p8[0], p8[1], p8[2], p8[3], p8[4], p8[5], yaw], dtype=np.float32)