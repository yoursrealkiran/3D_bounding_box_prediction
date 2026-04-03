import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes
import argparse

def show_image_with_mask_and_bbox(image_path: str, mask_path: str) -> None:
    # Read image and mask 
    image = read_image(image_path)
    masks_np = np.load(mask_path).astype(np.int8)
    masks = torch.from_numpy(masks_np)
    
    # masks_to_boxes expects a boolean tensor [N, H, W]
    # We also ensure it's a bool type here
    boxes_2d = masks_to_boxes(masks.bool())
    
    # boxes_2d is ALREADY a tensor, so just cast it to int
    boxes_2d = boxes_2d.to(dtype=torch.int)
    
    # Draw mask on image 
    image_with_masks = draw_segmentation_masks(image, masks.bool(), alpha=0.5, colors="blue")
    
    # Draw bounding boxes on image 
    image_with_masks_and_boxes = draw_bounding_boxes(image_with_masks, boxes_2d, colors="red", width=2)
    
    # Convert tensor to PIL image
    img_pil = F.to_pil_image(image_with_masks_and_boxes)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(np.asarray(img_pil))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show image with mask and 2D bounding box")
    parser.add_argument("image_path", type=str, help="image path")
    parser.add_argument("mask_path", type=str, help="mask path")
    
    args = parser.parse_args()
    
    show_image_with_mask_and_bbox(args.image_path, args.mask_path)