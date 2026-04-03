import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from configs.load_config import load_config
from models.pipeline_main import Fusion3DDetector
from utils.geometry import params_8_to_params_7
from utils.visualizer import draw_projected_box

def run_evaluation(checkpoint_path, scene_dir):
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # K matrix for 640x480 resolution
    K = np.array([[615, 0, 320], [0, 615, 240], [0, 0, 1]], dtype=np.float32)
    
    model = Fusion3DDetector()
    
    # 1. LOAD CHECKPOINT
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("model.", "") if k.startswith("model.") else k
        new_state_dict[name] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()

    os.makedirs("results", exist_ok=True)
    scenes = sorted([d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))])
    
    print(f"Starting Evaluation with Corrected NMS Logic...")

    for scene in scenes:
        path = os.path.join(scene_dir, scene)
        rgb_raw = np.array(Image.open(os.path.join(path, "rgb.jpg")))
        pc_raw = np.load(os.path.join(path, "pc.npy")).astype(np.float32).copy()
        
        # 2. PREPROCESSING
        pc_proc = pc_raw.copy()
        pc_proc[0, :, :] = -pc_proc[0, :, :] # Flip X
        pc_proc[1, :, :] = -pc_proc[1, :, :] # Flip Y
        pc_proc[2, :, :] -= 1.5
        pc_proc /= 2.0
        
        rgb_t = torch.from_numpy(rgb_raw).permute(2, 0, 1).float().div(255).unsqueeze(0)
        pc_t = torch.from_numpy(pc_proc).unsqueeze(0)

        rgb_t = F.interpolate(rgb_t, size=(480, 640), mode='bilinear', align_corners=False).to(device)
        pc_t = F.interpolate(pc_t, size=(480, 640), mode='nearest').to(device)

        with torch.no_grad():
            logits, bboxes = model(rgb_t, pc_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_maps = bboxes.squeeze().cpu().numpy()

        # 3. THRESHOLDING
        threshold = 0.4 
        y_idxs, x_idxs = np.where(probs > threshold)
        
        detections = []
        for y, x in zip(y_idxs, x_idxs):
            p8 = pred_maps[:, y, x]
            p7 = params_8_to_params_7(p8)
            
            # REVERSE NORMALIZATION & ALIGNMENT
            p7[0:6] *= 2.0
            p7[2] += 1.5
            p7[0] = -p7[0] 
            p7[1] = -p7[1] 
            
            # FIX: Ensure 'grid_pos' is added here!
            detections.append({
                'score': probs[y, x], 
                'params': p7, 
                'grid_pos': (y, x) # This is what was missing
            })
        
        # 4. SPATIAL NMS (Grid-Based)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        final_p7s = []
        drawn_grid_cells = []
        
        for det in detections:
            curr_grid = det['grid_pos'] # Now this will work!
            
            # If this cell is within 1.5 units of an already drawn peak, skip it
            if any(np.linalg.norm(np.array(curr_grid) - np.array(prev)) < 1.5 for prev in drawn_grid_cells):
                continue
            
            final_p7s.append(det['params'])
            drawn_grid_cells.append(curr_grid)

        # 5. VISUALIZATION
        vis_img = rgb_raw.copy()
        for p7 in final_p7s:
            vis_img = draw_projected_box(vis_img, p7, K, color=(0, 255, 0))

        save_path = os.path.join("results", f"{scene}_3d_detect.png")
        Image.fromarray(vis_img).save(save_path)
        print(f"Scene: {scene} | Detected: {len(final_p7s)}")

if __name__ == "__main__":
    ckpt = "checkpoints/3d-fusion-epoch=59-val_loss=0.00.ckpt"
    data_path = "/home/kiranraj-muthuraj/self_projects/3D_bounding_box_prediction/dl_challenge"
    run_evaluation(checkpoint_path=ckpt, scene_dir=data_path)