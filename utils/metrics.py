import torch

def get_3d_iou(box1_7d, box2_7d):
    """
    Calculates 3D IoU (Axis-Aligned Approximation for training speed).
    box1_7d: [x,y,z,w,l,h,yaw]
    """
    # Calculate AABB bounds
    # Assuming index 3=w, 4=l, 5=h
    def get_bounds(box):
        pos = box[:3]
        dims = box[3:6]
        return pos - dims/2, pos + dims/2

    b1_min, b1_max = get_bounds(box1_7d)
    b2_min, b2_max = get_bounds(box2_7d)

    inter_min = torch.max(b1_min, b2_min)
    inter_max = torch.min(b1_max, b2_max)
    
    inter_dims = torch.clamp(inter_max - inter_min, min=0)
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]

    vol1 = box1_7d[3] * box1_7d[4] * box1_7d[5]
    vol2 = box2_7d[3] * box2_7d[4] * box2_7d[5]
    
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / (union_vol + 1e-7)