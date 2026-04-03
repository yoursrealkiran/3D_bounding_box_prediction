import numpy as np

def params_8_to_params_7(p8):
    """8D [x,y,z,w,l,h,sin,cos] -> 7D [x,y,z,w,l,h,yaw]"""
    # atan2(sin, cos) is robust to all quadrants
    yaw = np.arctan2(p8[6], p8[7])
    return np.array([p8[0], p8[1], p8[2], p8[3], p8[4], p8[5], yaw], dtype=np.float32)

def params_7_to_corners_8(params_7):
    """7D [x,y,z,w,l,h,yaw] -> (8, 3) corners"""
    x, y, z, w, l, h, yaw = params_7
    
    # 1. Define corners in local object coordinates
    # We define: l=length(x-axis), h=height(y-axis), w=width(z-axis)
    x_c = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_c = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_c = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    corners = np.vstack([x_c, y_c, z_c])

    # 2. Rotation matrix (Yaw around Y-axis)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], 
                  [0, 1, 0], 
                  [-s, 0, c]])
    
    # 3. Transform to world coordinates
    corners_3d = np.dot(R, corners)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    return corners_3d.T