import numpy as np


def _safe_float32_array(arr, expected_last_dim=None):
    """
    Convert input to float32 numpy array and optionally validate last dimension.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if expected_last_dim is not None and arr.shape[-1] != expected_last_dim:
        raise ValueError(f"Expected last dimension {expected_last_dim}, got shape {arr.shape}")
    return arr


def yaw_from_sin_cos(sin_yaw, cos_yaw):
    """
    Recover yaw from sin/cos representation.
    """
    return np.arctan2(sin_yaw, cos_yaw).astype(np.float32)


def normalize_sin_cos(sin_cos, eps=1e-6):
    """
    Normalize a [sin, cos] vector.
    """
    sin_cos = np.asarray(sin_cos, dtype=np.float32)
    norm = np.linalg.norm(sin_cos)
    if norm < eps:
        return np.array([0.0, 1.0], dtype=np.float32)
    return (sin_cos / norm).astype(np.float32)


def params_8_to_params_7(p8):
    """
    8D [x, y, z, w, l, h, sin, cos] -> 7D [x, y, z, w, l, h, yaw]
    """
    p8 = _safe_float32_array(p8, expected_last_dim=8).copy()
    p8[6:8] = normalize_sin_cos(p8[6:8])
    yaw = yaw_from_sin_cos(p8[6], p8[7])
    return np.array([p8[0], p8[1], p8[2], p8[3], p8[4], p8[5], yaw], dtype=np.float32)


def params_7_to_params_8(p7):
    """
    7D [x, y, z, w, l, h, yaw] -> 8D [x, y, z, w, l, h, sin, cos]
    """
    p7 = _safe_float32_array(p7, expected_last_dim=7)
    x, y, z, w, l, h, yaw = p7
    return np.array([x, y, z, w, l, h, np.sin(yaw), np.cos(yaw)], dtype=np.float32)


def params_7_to_corners_8(params_7):
    """
    7D [x, y, z, w, l, h, yaw] -> (8, 3) corners

    Convention used:
    - l = local x-axis extent
    - h = local y-axis extent
    - w = local z-axis extent
    - yaw rotates around global Y-axis
    """
    params_7 = _safe_float32_array(params_7, expected_last_dim=7)
    x, y, z, w, l, h, yaw = params_7

    # Local object corners
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float32)
    y_c = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2], dtype=np.float32)
    z_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float32)
    corners = np.vstack([x_c, y_c, z_c])  # [3, 8]

    c = np.cos(yaw).astype(np.float32)
    s = np.sin(yaw).astype(np.float32)
    R = np.array(
        [[ c, 0.0,  s],
         [0.0, 1.0, 0.0],
         [-s, 0.0,  c]],
        dtype=np.float32,
    )

    corners_3d = R @ corners
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z

    return corners_3d.T.astype(np.float32)


def world_to_grid(center_x, center_z, grid_size=(15, 20), range_x=(-10.0, 10.0), range_z=(0.0, 20.0)):
    """
    Convert world coordinates (x, z) to continuous grid coordinates.

    Returns:
        gx_f, gz_f
    """
    grid_h, grid_w = grid_size
    x_min, x_max = range_x
    z_min, z_max = range_z

    gx_f = (center_x - x_min) / max(x_max - x_min, 1e-6) * grid_w
    gz_f = (center_z - z_min) / max(z_max - z_min, 1e-6) * grid_h
    return float(gx_f), float(gz_f)


def grid_to_world(grid_x, grid_z, grid_size=(15, 20), range_x=(-10.0, 10.0), range_z=(0.0, 20.0)):
    """
    Convert continuous grid coordinates back to world coordinates.
    """
    grid_h, grid_w = grid_size
    x_min, x_max = range_x
    z_min, z_max = range_z

    world_x = x_min + (grid_x / grid_w) * (x_max - x_min)
    world_z = z_min + (grid_z / grid_h) * (z_max - z_min)
    return float(world_x), float(world_z)


def params_10_to_params_7(
    p10,
    grid_pos,
    grid_size=(15, 20),
    range_x=(-10.0, 10.0),
    range_z=(0.0, 20.0),
):
    """
    Convert model output:
      [x, y, z, w, l, h, sin, cos, off_x, off_z]
    into:
      [x, y, z, w, l, h, yaw]

    Notes:
    - x and z are reconstructed from (grid_pos + offsets).
    - y is taken directly from regression output.
    - p10[0] and p10[2] are ignored by design in this dense-grid formulation.
    """
    p10 = _safe_float32_array(p10, expected_last_dim=10).copy()
    p10[6:8] = normalize_sin_cos(p10[6:8])

    gy, gx = int(grid_pos[0]), int(grid_pos[1])
    off_x = float(p10[8])
    off_z = float(p10[9])

    gx_f = gx + off_x
    gz_f = gy + off_z

    world_x, world_z = grid_to_world(
        gx_f,
        gz_f,
        grid_size=grid_size,
        range_x=range_x,
        range_z=range_z,
    )

    yaw = yaw_from_sin_cos(p10[6], p10[7])

    return np.array(
        [world_x, p10[1], world_z, p10[3], p10[4], p10[5], yaw],
        dtype=np.float32,
    )


def corners_8_to_center(corners):
    """
    Compute box center from 8 corners.
    """
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (8, 3):
        raise ValueError(f"Expected corners shape (8, 3), got {corners.shape}")
    return np.mean(corners, axis=0).astype(np.float32)