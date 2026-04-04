import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from configs.load_config import load_config
from models.pipeline_main import Fusion3DDetector
from utils.geometry import params_10_to_params_7 
from utils.visualizer import draw_projected_box

def run_evaluation(checkpoint_path, scene_dir):
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Camera intrinsics (Standard for 640x480 resolution)
    K = np.array([[615, 0, 320], [0, 615, 240], [0, 0, 1]], dtype=np.float32)
    
    model = Fusion3DDetector()
    
    # 1. LOAD CHECKPOINT
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Clean up Lightning prefix if necessary
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[name] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()

    os.makedirs("results", exist_ok=True)
    scenes = sorted([d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))])
    
    grid_h, grid_w = tuple(cfg['data']['grid_size'])

    print(f"Starting Evaluation on {len(scenes)} scenes...")

    for scene in scenes:
        path = os.path.join(scene_dir, scene)
        rgb_raw = np.array(Image.open(os.path.join(path, "rgb.jpg")))
        pc_raw = np.load(os.path.join(path, "pc.npy")).astype(np.float32).copy()
        
        # 2. PREPROCESSING 
        pc_proc = pc_raw.copy()
        # Rotate 180 on Z to align with Camera Forward
        pc_proc[0, :, :] = -pc_proc[0, :, :] 
        pc_proc[1, :, :] = -pc_proc[1, :, :] 
        
       
        pc_proc[2, :, :] -= 1.5
        
        rgb_t = torch.from_numpy(rgb_raw).permute(2, 0, 1).float().div(255).unsqueeze(0)
        pc_t = torch.from_numpy(pc_proc).unsqueeze(0)
        
        # Use bilinear for RGB and nearest for PC to maintain geometry
        rgb_t = F.interpolate(rgb_t, size=(480, 640), mode='bilinear', align_corners=False).to(device)
        pc_t = F.interpolate(pc_t, size=(480, 640), mode='nearest').to(device)

        with torch.no_grad():
            logits, bboxes = model(rgb_t, pc_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_maps = bboxes.squeeze().cpu().numpy() # Shape [10, 15, 20]

        # 3. THRESHOLDING & PARAMETER RECOVERY
        threshold = 0.5 # Increased slightly to reduce noise
        y_idxs, x_idxs = np.where(probs > threshold)
        
        detections = []
        for y, x in zip(y_idxs, x_idxs):
            p10 = pred_maps[:, y, x]
            
            # STEP A: Reconstruct world coordinates using high-precision offsets
            # This turns the discrete grid index into a continuous world coordinate
            p7 = params_10_to_params_7(p10, (y, x), grid_size=(grid_h, grid_w))
            
            # STEP B: REVERSE DEPTH SHIFT (Only undoing the -1.5)
            # p7[0:6] scaling removed to match updated dataset.py
            p7[2] += 1.5
            
            # STEP C: REVERSE ROTATION (Undo X/Y flips)
            p7[0] = -p7[0] 
            p7[1] = -p7[1] 
            
            # DEBUG LOG: Check if Z is varying or stuck
            if len(detections) < 1: # Only print first detection per scene to keep logs clean
                print(f"Scene {scene} Peak: Grid({y},{x}) Score: {probs[y,x]:.2f} World Z: {p7[2]:.2f}")

            detections.append({
                'score': probs[y, x], 
                'params': p7, 
                'grid_pos': (y, x) 
            })
        
        # 4. SPATIAL NMS (Grid-Based)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        final_p7s = []
        drawn_grid_cells = []
        
        for det in detections:
            curr_grid = det['grid_pos']
            
            if any(np.linalg.norm(np.array(curr_grid) - np.array(prev)) < 1.5 for prev in drawn_grid_cells):
                continue
            final_p7s.append(det['params'])
            drawn_grid_cells.append(curr_grid)

        # 5. VISUALIZATION
        vis_img = rgb_raw.copy()
        for p7 in final_p7s:
            # Wireframe 3D box projected onto the 2D image
            vis_img = draw_projected_box(vis_img, p7, K, color=(0, 255, 0))

        save_path = os.path.join("results", f"{scene}_3d_detect.png")
        Image.fromarray(vis_img).save(save_path)

if __name__ == "__main__":
    
    ckpt = "checkpoints/fusion-3d-epoch=84-val_iou=0.00.ckpt"
    data_path = "/home/kiranraj-muthuraj/self_projects/3D_bounding_box_prediction/dl_challenge"
    run_evaluation(checkpoint_path=ckpt, scene_dir=data_path)