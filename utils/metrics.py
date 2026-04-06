import torch


def _to_tensor(box):
    """
    Convert input to float tensor.
    """
    if not isinstance(box, torch.Tensor):
        box = torch.tensor(box, dtype=torch.float32)
    return box.float()


def get_aabb_bounds(box_7d):
    """
    Convert [x, y, z, w, l, h, yaw] into axis-aligned min/max bounds.

    Note:
    - yaw is ignored here
    - this is an AABB approximation
    """
    box_7d = _to_tensor(box_7d)
    center = box_7d[:3]
    dims = torch.clamp(box_7d[3:6], min=1e-6)

    half = dims / 2.0
    bmin = center - half
    bmax = center + half
    return bmin, bmax


def get_3d_iou(box1_7d, box2_7d):
    """
    Axis-aligned 3D IoU approximation for:
      [x, y, z, w, l, h, yaw]

    This ignores yaw and is mainly for quick validation.
    """
    b1_min, b1_max = get_aabb_bounds(box1_7d)
    b2_min, b2_max = get_aabb_bounds(box2_7d)

    inter_min = torch.max(b1_min, b2_min)
    inter_max = torch.min(b1_max, b2_max)

    inter_dims = torch.clamp(inter_max - inter_min, min=0.0)
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]

    box1_7d = _to_tensor(box1_7d)
    box2_7d = _to_tensor(box2_7d)

    vol1 = torch.clamp(box1_7d[3], min=1e-6) * torch.clamp(box1_7d[4], min=1e-6) * torch.clamp(box1_7d[5], min=1e-6)
    vol2 = torch.clamp(box2_7d[3], min=1e-6) * torch.clamp(box2_7d[4], min=1e-6) * torch.clamp(box2_7d[5], min=1e-6)

    union_vol = vol1 + vol2 - inter_vol
    iou = inter_vol / torch.clamp(union_vol, min=1e-7)
    return iou


def center_distance(box1_7d, box2_7d):
    """
    Euclidean distance between box centers.
    """
    box1_7d = _to_tensor(box1_7d)
    box2_7d = _to_tensor(box2_7d)
    return torch.norm(box1_7d[:3] - box2_7d[:3], p=2)


def size_l1_error(box1_7d, box2_7d):
    """
    Mean absolute error on box dimensions [w, l, h].
    """
    box1_7d = _to_tensor(box1_7d)
    box2_7d = _to_tensor(box2_7d)
    return torch.mean(torch.abs(box1_7d[3:6] - box2_7d[3:6]))


def yaw_error_rad(box1_7d, box2_7d):
    """
    Smallest absolute yaw difference in radians.
    """
    box1_7d = _to_tensor(box1_7d)
    box2_7d = _to_tensor(box2_7d)

    yaw1 = box1_7d[6]
    yaw2 = box2_7d[6]

    diff = yaw1 - yaw2
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.abs(diff)