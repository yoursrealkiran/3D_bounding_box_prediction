import torch
import onnx
from models.pipeline_main import Fusion3DDetector
from configs.load_config import load_config

def export_model(checkpoint_path, output_path="model.onnx"):
    # 1. Load Config and Model
    cfg = load_config("configs/config.yaml")
    model = Fusion3DDetector()
    
    # 2. Load Weights
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Strip 'model.' prefix if saved via Lightning
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Create Dummy Input
    # Match the shapes from your config (Batch, Channels, H, W)
    dummy_rgb = torch.randn(1, 3, 481, 607)
    dummy_pc = torch.randn(1, 3, 481, 607)

    # 4. Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        (dummy_rgb, dummy_pc), 
        output_path,
        export_params=True,        # Store the trained parameter weights inside the model file
        opset_version=11,          # Standard opset for broad compatibility
        do_constant_folding=True,  # Optimizes constants for faster inference
        input_names=['rgb_input', 'pc_input'],
        output_names=['cls_output', 'box_output'],
        dynamic_axes={             # Allow for variable batch sizes during deployment
            'rgb_input': {0: 'batch_size'},
            'pc_input': {0: 'batch_size'},
            'cls_output': {0: 'batch_size'},
            'box_output': {0: 'batch_size'}
        }
    )

    # 5. Verify the Export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model exported and verified successfully!")

if __name__ == "__main__":
    # Replace with your actual best checkpoint path
    export_model(
        checkpoint_path="checkpoints/best-bbox-model-epoch=28-val_iou=0.00.ckpt", 
        output_path="checkpoints/fusion_bbox_model.onnx"
    )