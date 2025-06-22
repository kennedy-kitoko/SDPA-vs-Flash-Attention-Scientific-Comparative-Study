# Experimental Methodology

## Research Design

This study employs a controlled experimental design to compare PyTorch's native SDPA with Flash Attention 2 under identical conditions.

## Dataset: Weeds-3

### Dataset Acquisition
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("augmented-startups").project("weeds-nxe1w")
version = project.version(3)
dataset = version.download("yolov11")  # Compatible with YOLOv12
```

### Dataset Characteristics
- **Type**: Agricultural weed detection
- **Classes**: 3 weed species
- **Training Images**: 3,664
- **Validation Images**: 359
- **Image Size**: 640x640 pixels
- **Annotations**: Bounding boxes in YOLO format

## Experimental Protocol

### 1. Environment Setup
- **Hardware**: NVIDIA RTX 4060 Laptop GPU (8GB VRAM)
- **Software**: PyTorch 2.2.2+cu118, CUDA 11.8
- **OS**: WSL2 Ubuntu on Windows

### 2. Model Configuration
- **Architecture**: YOLOv12n (2.56M parameters)
- **Attention Blocks**: A2C2f modules (layers 6, 8, 11, 14, 17)
- **Input Size**: 640x640
- **Batch Size**: 8 (memory-constrained)

### 3. Training Parameters
```python
config = {
    'epochs': 20,
    'batch': 8,
    'imgsz': 640,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'patience': 10,
    'deterministic': True,
    'seed': 0
}
```

### 4. Attention Configuration

#### SDPA (PyTorch Native)
```python
os.environ['TORCH_CUDNN_BENCHMARK'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Automatic selection of best SDPA backend
```

#### Flash Attention 2
```python
os.environ['FLASH_ATTENTION_FORCE_USE'] = '1'
# Requires pre-compiled flash-attn package
```

### 5. Metrics Collection
- **Training Time**: Wall-clock time per epoch
- **Memory Usage**: Peak GPU memory allocation
- **Model Performance**: mAP50, mAP50-95
- **Convergence**: Loss curves over epochs

## Reproducibility

### Random Seed Control
```python
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
```

### Data Augmentation
Consistent augmentation pipeline using Albumentations:
- Blur (p=0.01)
- MedianBlur (p=0.01)
- ToGray (p=0.01)
- CLAHE (p=0.01)

## Statistical Validity

- **Sample Size**: 20 epochs provide sufficient convergence
- **Repeated Measures**: Identical configuration ensures comparability
- **Control Variables**: Hardware, software, data, hyperparameters

## Limitations

1. **Single GPU Type**: Results may vary on different architectures
2. **Dataset Specificity**: Agricultural domain may not generalize
3. **Model Size**: Only tested on nano variant
4. **Batch Size**: Limited by GPU memory

## Ethical Considerations

This research aims to democratize AI by reducing technical barriers, particularly benefiting researchers in resource-constrained environments.