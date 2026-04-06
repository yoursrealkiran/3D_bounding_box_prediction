import numpy as np

def draw_msra_gaussian(heatmap, center, sigma):
    """
    Draw a 2D Gaussian onto a heatmap.

    Args:
        heatmap: (H, W) float32 array
        center: (grid_x, grid_y)
        sigma: gaussian sigma in grid cells
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
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
    )
    return heatmap


def corners_8_to_params_8(corners):
    """
    Convert 8 corners [8, 3] to [x, y, z, w, l, h, sin, cos].

    Assumes corner ordering is consistent across the dataset.
    """
    center = np.mean(corners, axis=0)

    w = np.linalg.norm(corners[0] - corners[4])
    l = np.linalg.norm(corners[0] - corners[3])
    h = np.linalg.norm(corners[0] - corners[1])

    direction_vector = corners[3] - corners[0]
    yaw = np.arctan2(direction_vector[2], direction_vector[0])

    return np.array(
        [center[0], center[1], center[2], w, l, h, np.sin(yaw), np.cos(yaw)],
        dtype=np.float32,
    )


def world_to_grid(center_x, center_z, grid_size, range_x, range_z):
    """
    Map world coordinates (x, z) to continuous grid coordinates.

    Returns:
        gx_f, gz_f : floating point grid coordinates
    """
    grid_h, grid_w = grid_size
    x_min, x_max = range_x
    z_min, z_max = range_z

    gx_f = (center_x - x_min) / max(x_max - x_min, 1e-6) * grid_w
    gz_f = (center_z - z_min) / max(z_max - z_min, 1e-6) * grid_h
    return gx_f, gz_f


def grid_to_world(grid_x, grid_z, grid_size, range_x, range_z):
    """
    Map continuous grid coordinates back to world coordinates.
    """
    grid_h, grid_w = grid_size
    x_min, x_max = range_x
    z_min, z_max = range_z

    world_x = x_min + (grid_x / grid_w) * (x_max - x_min)
    world_z = z_min + (grid_z / grid_h) * (z_max - z_min)
    return world_x, world_z


def params_10_to_params_7(p10, grid_pos, grid_size=(15, 20), range_x=(-10.0, 10.0), range_z=(0.0, 20.0)):
    """
    Convert model output [x, y, z, w, l, h, sin, cos, off_x, off_z]
    into [x, y, z, w, l, h, yaw] in world coordinates.

    Here x and z are reconstructed from grid position + offsets.
    """
    yaw = np.arctan2(p10[6], p10[7])

    gy, gx = grid_pos
    off_x, off_z = p10[8], p10[9]

    gx_f = gx + off_x
    gz_f = gy + off_z

    world_x, world_z = grid_to_world(
        gx_f,
        gz_f,
        grid_size=grid_size,
        range_x=range_x,
        range_z=range_z,
    )

    return np.array(
        [world_x, p10[1], world_z, p10[3], p10[4], p10[5], yaw],
        dtype=np.float32,
    )


def params_8_to_params_7(p8):
    """
    Fallback for [x, y, z, w, l, h, sin, cos] -> [x, y, z, w, l, h, yaw]
    """
    yaw = np.arctan2(p8[6], p8[7])
    return np.array([p8[0], p8[1], p8[2], p8[3], p8[4], p8[5], yaw], dtype=np.float32)