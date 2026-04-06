import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from data.dataset import ThreeDObjectDataset

save_dir = "debug_heatmaps_gt"
os.makedirs(save_dir, exist_ok=True)

ds = ThreeDObjectDataset(
    root_dir="/home/kiranraj-muthuraj/self_projects/3D_bounding_box_prediction/dl_challenge",
    target_size=(480, 640),
    grid_size=(15, 20),
    training=False,
    range_x=(-0.3, 0.3),
    range_z=(-0.8, 0.0),
    gaussian_sigma=1.0,
    z_shift=1.5,
)

for i in range(5):
    sample = ds[i]

    rgb = sample["rgb"].permute(1, 2, 0).numpy()
    heatmap = sample["target_map"][0].numpy()

    # Approximate center peaks for visualization
    peak_ys, peak_xs = np.where(heatmap >= 0.95)

    # Save RGB
    plt.figure(figsize=(8, 5))
    plt.imshow(rgb)
    plt.title(f"RGB {i}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"rgb_{i}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save heatmap in default matrix view
    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar()
    plt.title(f"GT Heatmap {i} (Matrix View)")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Z")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gt_heatmap_{i}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save heatmap in BEV-style view
    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap, cmap="hot", origin="lower")
    plt.colorbar()
    if len(peak_xs) > 0:
        plt.scatter(peak_xs, peak_ys, c="cyan", s=25, marker="o", label="Peak cells")
        plt.legend(loc="upper right")
    plt.title(f"GT Heatmap {i} (BEV View)")
    plt.xlabel("X grid")
    plt.ylabel("Z grid")
    plt.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"gt_heatmap_bev_{i}.png"), dpi=150, bbox_inches="tight")
    plt.close()

print(f"Saved heatmap visualizations to: {save_dir}")