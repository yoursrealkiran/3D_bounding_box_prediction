import torch
import open3d as o3d
import lightning as L

print(f"PyTorch Version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"Lightning Version: {L.__version__}")
print(f"Open3D Version: {o3d.__version__}")