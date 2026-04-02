import matplotlib.pyplot as plt
import numpy as np
import cv2
from .geometry import params_7_to_corners_8 

def plot_bev_with_boxes(pc, gt_box, pred_box, save_path):
    """
    Creates a Bird's Eye View (X-Z plane) plot.
    pc: (3, H, W)
    boxes: [x, y, z, w, h, l, yaw]
    """
    plt.figure(figsize=(8, 8))
    
    # Flatten Point Cloud for scattering
    x_points = pc[0].flatten()
    z_points = pc[2].flatten()
    
    # Subsample for speed
    idx = np.random.choice(len(x_points), 5000, replace=False)
    plt.scatter(x_points[idx], z_points[idx], s=1, c='gray', alpha=0.3)
    
    # Plot GT Box (Green)
    gt_corners = params_7_to_corners_8(gt_box)
    plt.plot(gt_corners[[0,1,2,3,0], 0], gt_corners[[0,1,2,3,0], 2], 'g-', label='GT')
    
    # Plot Pred Box (Red)
    pred_corners = params_7_to_corners_8(pred_box)
    plt.plot(pred_corners[[0,1,2,3,0], 0], pred_corners[[0,1,2,3,0], 2], 'r-', label='Pred')
    
    plt.legend()
    plt.title("Bird's Eye View (BEV) Prediction vs Ground Truth")
    plt.savefig(save_path)
    plt.close()