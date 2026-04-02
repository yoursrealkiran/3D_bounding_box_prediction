import torch
import numpy as np

def get_3d_iou(box1, box2):
    """
    Calculates 3D IoU for two boxes.
    Input: [x, y, z, w, h, l, yaw]
    """
    def get_axis_aligned_coords(box):
        x, y, z, w, h, l, _ = box
        return (x - w/2, y - h/2, z - l/2), (x + w/2, y + h/2, z + l/2)

    # Convert to min/max corners (AABB approximation)
    min1, max1 = get_axis_aligned_coords(box1)
    min2, max2 = get_axis_aligned_coords(box2)

    # Find Intersection
    inter_min = torch.max(torch.tensor(min1), torch.tensor(min2))
    inter_max = torch.min(torch.tensor(max1), torch.tensor(max2))
    
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0))

    # Find Union
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]
    union_vol = vol1 + vol2 - inter_vol

    return inter_vol / (union_vol + 1e-7)

def average_precision_3d(preds, targets, iou_threshold=0.5):
    """
    Computes if predictions are correct based on IoU threshold.
    """
    ious = [get_3d_iou(p, t) for p, t in zip(preds, targets)]
    correct = [1 if iou >= iou_threshold else 0 for iou in ious]
    return np.mean(correct)