import os
import torch
import onnx

from models.pipeline_main import Fusion3DDetector
from configs.load_config import load_config


def load_checkpoint_weights(model, checkpoint_path):
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k[len("model."):]] = v
        else:
            cleaned_state_dict[k] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")

    return model


def export_model(checkpoint_path, output_path="checkpoints/fusion_bbox_model.onnx"):
    # 1. Load config
    cfg = load_config("configs/config.yaml")

    target_h, target_w = tuple(cfg["data"]["target_size"])
    grid_h, grid_w = tuple(cfg["data"]["grid_size"])

    # 2. Build model exactly like training/eval
    model = Fusion3DDetector(
        freeze_rgb_layers=bool(cfg["model"].get("freeze_rgb_layers", True)),
        head_dropout=float(cfg["model"].get("dropout", 0.2)),
        output_grid_size=(grid_h, grid_w),
    )

    # 3. Load weights
    model = load_checkpoint_weights(model, checkpoint_path)
    model.eval()

    # 4. Dummy inputs
    # Use resized training/eval resolution, not raw scene size
    dummy_rgb = torch.randn(1, 3, target_h, target_w, dtype=torch.float32)
    dummy_pc = torch.randn(1, 3, target_h, target_w, dtype=torch.float32)

    # 5. Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 6. Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_pc),
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["rgb_input", "pc_input"],
        output_names=["cls_output", "box_output", "seg_output"],
        dynamic_axes={
            "rgb_input": {0: "batch_size"},
            "pc_input": {0: "batch_size"},
            "cls_output": {0: "batch_size"},
            "box_output": {0: "batch_size"},
            "seg_output": {0: "batch_size"},
        },
    )

    # 7. Verify exported ONNX
    print("Verifying exported ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print("ONNX model exported and verified successfully!")
    print(f"Input RGB shape  : [B, 3, {target_h}, {target_w}]")
    print(f"Input PC shape   : [B, 3, {target_h}, {target_w}]")
    print(f"Output CLS shape : [B, 1, {grid_h}, {grid_w}]")
    print(f"Output BOX shape : [B, 10, {grid_h}, {grid_w}]")
    print(f"Output SEG shape : [B, 1, {grid_h}, {grid_w}]")


if __name__ == "__main__":
    export_model(
        checkpoint_path="checkpoints/last.ckpt",
        output_path="checkpoints/fusion_bbox_model.onnx",
    )