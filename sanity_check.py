import torch
from data.data_module import ThreeDDataModule
from models.pipeline_main import Fusion3DDetector
from configs.load_config import load_config

def test_pipeline():
    print("--- Starting Sanity Check ---")
    cfg = load_config("configs/config.yaml")
    
    # 1. Test Data Loading
    dm = ThreeDDataModule(
        data_dir=cfg['data']['root_dir'],
        batch_size=2
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    
    # Get one batch
    batch = next(iter(train_loader))
    rgb = batch['rgb']
    pc = batch['pc']
    gt_box = batch['gt_box']
    
    print(f"Data Loaded Successfully!")
    print(f"RGB Shape: {rgb.shape}")  # Expected: [batch, 3, 481, 607]
    print(f"PC Shape:  {pc.shape}")   # Expected: [batch, 3, 481, 607]
    print(f"GT Box Shape: {gt_box.shape}") # Expected: [batch, 7]

    # 2. Test Range Check (Critical for 3D)
    print(f"\nCoordinate Ranges:")
    print(f"PC X range: {pc[:,0].min():.2f} to {pc[:,0].max():.2f}")
    print(f"PC Z range: {pc[:,2].min():.2f} to {pc[:,2].max():.2f}")
    print(f"GT X range: {gt_box[:,0].min():.2f} to {gt_box[:,0].max():.2f}")

    # 3. Test Model Forward Pass
    model = Fusion3DDetector()
    logits, pred_box = model(rgb, pc)
    
    print(f"\nModel Forward Pass Successful!")
    print(f"Pred Box Shape: {pred_box.shape}")
    print("--- Sanity Check Passed ---")

if __name__ == "__main__":
    test_pipeline()