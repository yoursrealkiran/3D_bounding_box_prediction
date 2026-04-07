# 3D Bounding Box Prediction using RGB–Point Cloud Fusion

This project implements a dense multi-modal 3D object detection pipeline for cluttered scenes using:

- RGB images
- organized point clouds
- dense heatmap-based center prediction
- 3D box regression

The model uses a two-stream architecture:
- an RGB backbone for semantic features
- a point cloud backbone for geometric features
- a fusion head for joint reasoning
- a dense prediction head for center heatmaps and 3D box parameters

---

## Project Structure

```bash
3D_bounding_box_prediction/
├── configs/
│   ├── __init__.py
│   ├── config.yaml
│   └── load_config.py
├── data/
│   ├── __init__.py
│   ├── data_module.py
│   ├── dataset.py
│   └── utils.py
├── models/
│   ├── __init__.py
│   ├── backbone_pc.py
│   ├── backbone_rgb.py
│   ├── fusion_node.py
│   └── pipeline_main.py
├── utils/
│   ├── __init__.py
│   ├── geometry.py
│   ├── metrics.py
│   ├── visualizer.py
│   └── heatmap_viz.py
├── checkpoints/
├── results/
├── train.py
├── eval.py
├── export.py
└── README.md

```

---

## Dataset Format

```bash
scene_xxx/
├── rgb.jpg
├── pc.npy
├── mask.npy
└── bbox3d.npy
```


---

## 🚀 Setup Instructions

### 1. Clone the Repository

In a terminal,

`git clone https://github.com/yoursrealkiran/3D_bounding_box_prediction.git`

`cd 3D_bounding_box_prediction`

### 2. Install uv

`curl -LsSf https://astral.sh/uv/install.sh | sh`

`uv --version`

### 3. Create a Virual Environment

`uv venv`

#### Activate the environment

`source .venv/bin/activate`

### 4. Install Dependencies

`uv sync`

### 5. Run training

Set the `root_dir` in config.yaml file to dataset folder path

`uv run python train.py`

### 5. Run inference (evaluation) 

`uv run python eval.py`

### 6. Export into ONNX format

To export the trained model into the ONNX format for deployment and inference in production environments.

`uv run python export.py`
