import torch
import numpy as np
import os
import lightning as L
from PIL import Image
from configs.load_config import load_config
from models.pipeline_main import Fusion3DDetector
from utils.visualizer import plot_bev_with_boxes
from utils.metrics import get_3d_iou

def run_evaluation(checkpoint_path, scene_dir):
    # 1. Load Config and Model
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture and load weights
    # Note: If using Lightning, weights are inside 'state_dict'
    model = Fusion3DDetector()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both Lightning checkpoints and raw state_dicts
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Remove 'model.' prefix if saved via Lightning
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # 2. Prepare Results Directory
    os.makedirs("results", exist_ok=True)

    # 3. Inference Loop
    scenes = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]
    all_ious = []

    print(f"Starting Evaluation on {len(scenes)} scenes...")

    for scene in scenes:
        path = os.path.join(scene_dir, scene)
        
        # Load Data
        rgb = np.array(Image.open(f"{path}/rgb.jpg")) / 255.0
        pc = np.load(f"{path}/pc.npy")
        gt_box_raw = np.load(f"{path}/bbox3d.npy")[0] # Ground truth 8 corners
        
        # Convert to Tensors
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
        pc_t = torch.from_numpy(pc).float().unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits, pred_box_7d = model(rgb_t, pc_t)
            pred_box_7d = pred_box_7d.squeeze().cpu()

        # 4. Metrics & Visualization
        from data.utils import corners_8_to_params_7
        gt_box_7d = torch.from_numpy(corners_8_to_params_7(gt_box_raw))
        
        iou = get_3d_iou(pred_box_7d, gt_box_7d)
        all_ious.append(iou.item())

        # Save BEV Visualization
        plot_bev_with_boxes(
            pc, 
            gt_box_7d.numpy(), 
            pred_box_7d.numpy(), 
            f"results/{scene}_bev.png"
        )
        
        print(f"Scene: {scene} | 3D IoU: {iou.item():.4f}")

    print("-" * 30)
    print(f"Mean 3D IoU: {np.mean(all_ious):.4f}")
    print(f"Results and BEV plots saved to ./results/")

if __name__ == "__main__":
    
    run_evaluation(
        checkpoint_path="checkpoints/best-bbox-model-epoch=28-val_iou=0.00.ckpt", 
        scene_dir="/home/kiranraj-muthuraj/Jobs/sereact/Imitation Learning Researcher/take‑home assignment/dataset/dl_challenge"
    )