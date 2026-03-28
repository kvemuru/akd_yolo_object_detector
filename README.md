# Akida YOLO Object Detector

A YOLOv2-based object detection model targeted for the Akida 1000 neuromorphic processor using TensorFlow/Keras and BrainChip's MetaTF framework.

## Overview

This project implements an efficient object detector that can:
- Detect 20 PASCAL-VOC object classes
- Run on Akida 1000 neuromorphic hardware
- Be trained, pruned, quantized, and converted for deployment


## Features

- **Akida 1.0 Compatible**: All layers follow hardware constraints (8-bit weights, 4-bit activations)
- **YOLOv2 Architecture**: Single-shot object detection with anchor boxes
- **AkidaNet Backbone**: Efficient separable convolution backbone
- **Full Pipeline Support**: Train → Prune → Quantize → Convert → Infer
- **VOC Dataset Support**: Pre-built support for PASCAL-VOC 2007 dataset

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd akd_object_detector

# Create conda environment
conda env create -f environment.yml

# Or install dependencies directly
pip install -r requirements.txt
```

### Activate Environment

```bash
conda activate tf_akida
```

## Quick Start

### Training with VOC Dataset (5 epochs)

```bash
cd training
python train_voc.py
```

This will:
1. Download VOC 2007 dataset (~450MB) if not present
2. Load 2,501 training images and 2,510 validation images
3. Train for 5 epochs
4. Output training loss and validation loss

### Expected Output

```
============================================================
TRAINING WITH VOC DATASET
============================================================

[1/4] Downloading VOC dataset...
VOC dataset already exists at ./data/VOCdevkit/VOC2007

[2/4] Loading training data...
Loaded 2501 training images

[3/4] Loading validation data...
Loaded 2510 validation images

[4/4] Creating model...
Model parameters: 353,709

============================================================
TRAINING
============================================================
Epoch 1/5 - Loss: 0.1152 - Val Loss: 0.0025
Epoch 2/5 - Loss: 0.0079 - Val Loss: 0.0025
Epoch 3/5 - Loss: 0.0037 - Val Loss: 0.0027
Epoch 4/5 - Loss: 0.0028 - Val Loss: 0.0030
Epoch 5/5 - Loss: 0.0027 - Val Loss: 0.0033
```

## CLI Usage

The main CLI provides multiple operation modes:

| Mode | Command | Description |
|------|---------|-------------|
| Train | `python main.py --mode train` | Train with dummy/generator data |
| Prune | `python main.py --mode prune` | Apply weight pruning |
| Quantize | `python main.py --mode quantize` | 8-bit quantization |
| Convert | `python main.py --mode convert` | Convert to Akida format |
| Infer | `python main.py --mode infer --image <path>` | Run inference on image |
| Full | `python main.py --mode full` | Run entire pipeline |

### Examples

```bash
# Train with custom config
python main.py --mode train --config config.yaml

# Run inference on an image
python main.py --mode infer --image test.jpg --weights checkpoints/model.weights.h5

# Full pipeline (train, prune, quantize, convert)
python main.py --mode full
```

## Configuration

Edit `config.yaml` to customize training:

```yaml
model:
  name: "akida_yolo_detector"
  input_shape: [224, 224, 3]
  num_classes: 20
  alpha: 0.5
  grid_size: [7, 7]
  num_anchors: 5

training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001

pruning:
  enabled: true
  target_sparsity: 0.7

quantization:
  enabled: true
  weights_bits: 8
  activations_bits: 4
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_shape` | Model input size (H, W, C) | [224, 224, 3] |
| `num_classes` | Number of object classes | 20 (VOC) |
| `grid_size` | YOLO grid dimensions | [7, 7] |
| `num_anchors` | Anchor boxes per grid cell | 5 |
| `alpha` | Network width multiplier | 0.5 |
| `epochs` | Training epochs | 5 |
| `batch_size` | Training batch size | 32 |
| `learning_rate` | Optimizer learning rate | 0.001 |

## Architecture

```
Input Image (224×224×3, 8-bit)
    │
    ▼
AkidaNet Backbone
    │
    ├── Conv2D (3×3, 16 filters)
    ├── SeparableConv2D (3×3, 32 filters)
    ├── SeparableConv2D (3×3, 64 filters)
    ├── ...
    └── Feature maps
    │
    ▼
YOLO Detection Head
    │
    ├── 5 anchor boxes per cell
    ├── 7×7 grid
    └── 25 outputs per anchor (4 bbox + 1 conf + 20 classes)
    │
    ▼
Output Tensor: (7, 7, 125)
```

### Output Format

For each of 7×7 grid cells and 5 anchors:
- **Bounding Box**: 4 values (tx, ty, tw, th)
- **Objectness Score**: 1 value (confidence)
- **Class Probabilities**: 20 values (one per VOC class)

Total: 7 × 7 × 5 × 25 = 6,125 values

### VOC Classes (20)

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | aeroplane | 10 | diningtable |
| 1 | bicycle | 11 | dog |
| 2 | bird | 12 | horse |
| 3 | boat | 13 | motorbike |
| 4 | bottle | 14 | person |
| 5 | bus | 15 | pottedplant |
| 6 | car | 16 | sheep |
| 7 | cat | 17 | sofa |
| 8 | chair | 18 | train |
| 9 | cow | 19 | tvmonitor |

## Project Structure

```
akd_object_detector/
├── models/
│   ├── akidanet.py       # AkidaNet backbone network
│   ├── yolo_head.py      # YOLO detection head
│   └── detector.py       # Combined detector model
├── training/
│   ├── train.py          # Main training loop
│   ├── train_voc.py      # VOC dataset training
│   └── losses.py         # YOLO loss functions
├── preprocessing/
│   └── utils.py         # Image preprocessing utilities
├── quantization/
│   ├── quantize.py       # 8-bit quantization
│   └── prune.py          # Weight pruning
├── conversion/
│   └── to_akida.py       # Akida model conversion
├── evaluation/
│   └── metrics.py        # mAP evaluation metrics
├── data/                  # Dataset storage
│   └── VOCdevkit/        # VOC dataset
├── checkpoints/          # Saved model weights
├── logs/                 # TensorBoard logs
├── config.yaml           # Configuration file
└── main.py              # CLI entry point
```

## Training Results

### 5-Epoch Training on VOC 2007

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1 | 0.1152 | 0.0025 |
| 2 | 0.0079 | 0.0025 |
| 3 | 0.0037 | 0.0027 |
| 4 | 0.0028 | 0.0030 |
| 5 | 0.0027 | 0.0033 |

- **Model Parameters**: 353,709
- **Trainable Parameters**: 349,549
- **Loss Reduction**: 98%
- **Training Time**: ~2 min/epoch (CPU)

## Hardware Constraints (Akida 1.0)

The model is designed to comply with Akida 1000 hardware constraints:

| Parameter | Constraint |
|-----------|------------|
| Input bitwidth | 8-bit |
| Weight bitwidth | 8-bit |
| Activation bitwidth | 1, 2, or 4-bit |
| Max kernel size | 7×7 |
| Stride-2 kernels | 3×3 only |

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'tensorflow'**
```bash
# Activate the correct conda environment
conda activate tf_akida
```

**2. VOC dataset download fails**
```python
# Manual download
from training.train_voc import download_voc_dataset
download_voc_dataset('./data')
```

**3. Out of memory errors**
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 16  # or 8
```

**4. ModelCheckpoint filepath error**
```python
# Ensure filepath ends with .weights.h5
filepath='weights.{epoch:02d}-{loss:.4f}.weights.h5'
```

## API Reference

### Training

```python
from training.train_voc import train_with_real_data, download_voc_dataset
import yaml

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Download dataset
download_voc_dataset('./data')

# Train
model, history = train_with_real_data(config, epochs=5)
```

### Model Creation

```python
from models.detector import create_detector

model = create_detector(
    input_shape=(224, 224, 3),
    num_classes=20,
    num_anchors=5,
    alpha=0.5
)
```

### Inference

```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess image
img = load_img('test.jpg', target_size=(224, 224))
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Run inference
predictions = model.predict(x)
```

## License

This project is for research and educational purposes.

## References

- [Akida Documentation](https://doc.brainchipinc.com/)
- [YOLOv2 Paper](https://arxiv.org/abs/1612.08242)
- [MetaTF Framework](https://github.com/brainchipinc)
- [PASCAL-VOC Dataset](http://host.robots.ox.ac.uk/pascal/voc/)
