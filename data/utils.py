import numpy as np
import torch

def draw_msra_gaussian(heatmap, center, sigma):
    """
    Draws a 2D Gaussian onto a heatmap.
    center: (int, int) -> (grid_x, grid_y)
    sigma: radius of the object in grid cells
    """
    tmp_size = 3 * sigma
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

    # Range check
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
    # Assuming standard order: 0-4 is width, 0-3 is length, 0-1 is height
    w = np.linalg.norm(corners[0] - corners[4])
    l = np.linalg.norm(corners[0] - corners[3])
    h = np.linalg.norm(corners[0] - corners[1])
    
    direction_vector = corners[3] - corners[0]
    yaw = np.arctan2(direction_vector[2], direction_vector[0])
    
    return np.array([center[0], center[1], center[2], w, l, h, np.sin(yaw), np.cos(yaw)], dtype=np.float32)

def params_8_to_params_7(p8):
    """[x, y, z, w, l, h, sin, cos] -> [x, y, z, w, l, h, yaw]"""
    # Recover yaw from sin/cos predicted by the model
    yaw = np.arctan2(p8[6], p8[7])
    return np.array([p8[0], p8[1], p8[2], p8[3], p8[4], p8[5], yaw], dtype=np.float32)