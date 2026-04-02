import numpy as np

def corners_8_to_params_7(corners):
    """
    Converts 8 corners (8, 3) to 7 parameters (x, y, z, w, h, l, yaw).
    Assumes corners are ordered: 0-3 bottom face, 4-7 top face.
    """
    # 1. Center is the mean of all vertices
    center = np.mean(corners, axis=0)
    
    # 2. Dimensions
    # Height (h): Distance between top and bottom faces
    h = np.linalg.norm(np.mean(corners[4:8], axis=0) - np.mean(corners[0:4], axis=0))
    # Width (w) and Length (l) from the bottom face
    w = np.linalg.norm(corners[0] - corners[1])
    l = np.linalg.norm(corners[1] - corners[2])
    
    # 3. Yaw (Rotation around vertical axis)
    # Vector from back-left to back-right corner
    direction_vector = corners[1] - corners[0]
    yaw = np.arctan2(direction_vector[2], direction_vector[0])
    
    return np.array([*center, w, h, l, yaw], dtype=np.float32)

def params_7_to_corners_8(params):
    """Useful for visualization: 7 params back to 8 corners"""
    x, y, z, w, h, l, yaw = params
    # Define a standard axis-aligned box at origin
    corners = np.array([
        [-w/2, -h/2, -l/2], [w/2, -h/2, -l/2], [w/2, -h/2, l/2], [-w/2, -h/2, l/2],
        [-w/2, h/2, -l/2], [w/2, h/2, -l/2], [w/2, h/2, l/2], [-w/2, h/2, l/2]
    ])
    # Rotate by yaw
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    corners = corners @ R.T
    # Translate to center
    corners += np.array([x, y, z])
    return corners