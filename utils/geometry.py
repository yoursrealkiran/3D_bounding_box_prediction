import numpy as np

def params_7_to_corners_8(params):
    """
    Input: [x, y, z, w, h, l, yaw]
    Output: (8, 3) corners
    """
    x, y, z, w, h, l, yaw = params
    
    # Define corners in local coordinates
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    
    # Rotation matrix (Yaw around Y-axis)
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # Rotate and translate
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    return corners_3d.T