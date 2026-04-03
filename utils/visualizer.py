import cv2
import numpy as np
import torch
from .geometry import params_8_to_params_7, params_7_to_corners_8

def project_3d_to_2d(corners_3d, K):
    """K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]"""
    # Prevent division by zero for points behind camera
    z = corners_3d[:, 2:3]
    z[z < 0.1] = 0.1 
    
    pts_2d = np.dot(K, corners_3d.T)
    pts_2d = pts_2d[:2, :] / pts_2d[2, :]
    return pts_2d.T

def draw_projected_box(image, params_7, K, color=(0, 255, 0)):
    corners_3d = params_7_to_corners_8(params_7)
    pts_2d = project_3d_to_2d(corners_3d, K).astype(int)

    # 12 lines of a cube
    edges = [
        (0,1), (1,2), (2,3), (3,0), # Bottom
        (4,5), (5,6), (6,7), (7,4), # Top
        (0,4), (1,5), (2,6), (3,7)  # Vertical
    ]

    for start, end in edges:
        cv2.line(image, tuple(pts_2d[start]), tuple(pts_2d[end]), color, 2)
    return image

def visualize_multi_objects(rgb_img, pred_map, K, threshold=0.6):
    """
    rgb_img: [H, W, 3] (0-1 float)
    pred_map: [9, 15, 20] (logits/heatmap + regression)
    """
    # Convert to 0-255 uint8 for CV2
    vis_img = (rgb_img * 255).astype(np.uint8).copy()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    heatmap = pred_map[0]
    # Sigmoid if the input is raw logits
    if heatmap.max() > 1.0 or heatmap.min() < 0.0:
        heatmap = 1 / (1 + np.exp(-heatmap))

    # Find peaks (Simple Local Maxima)
    indices = np.argwhere(heatmap > threshold)
    
    # Sort by confidence score (highest first)
    indices = sorted(indices, key=lambda i: heatmap[i[0], i[1]], reverse=True)

    drawn_centers = []
    for y, x in indices:
        # Simple NMS: Don't draw if center is too close to an existing box
        if any(np.linalg.norm(np.array([y,x]) - np.array(c)) < 1.5 for c in drawn_centers):
            continue
            
        params_8 = pred_map[1:, y, x]
        params_7 = params_8_to_params_7(params_8)
        
        # Draw Box
        vis_img = draw_projected_box(vis_img, params_7, K)
        drawn_centers.append((y, x))
        
    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)