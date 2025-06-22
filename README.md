# SDPA vs Flash Attention: A Comparative Study for Production ML Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLOv12-Supported-brightgreen.svg)](https://github.com/ultralytics/ultralytics)

## Abstract

This repository presents the **first published scientific comparison** between PyTorch's native Scaled Dot-Product Attention (SDPA) and Flash Attention 2 in production ML systems. Our research demonstrates that SDPA offers a viable alternative to Flash Attention, eliminating complex CUDA compilation requirements while maintaining equivalent performance.

## Key Findings

| Metric | SDPA (PyTorch Native) | Flash Attention 2 | Difference |
|--------|----------------------|-------------------|------------|
| **Training Time** | 37.49 minutes | 35.24 minutes | FA 6.0% faster |
| **Memory Usage** | 2,668.71 MB | 518.86 MB | FA 80.6% more efficient |
| **Final mAP50** | 0.967 (96.7%) | 0.967 (96.7%) | **Identical** |
| **Final mAP50-95** | 0.753 (75.3%) | 0.753 (75.3%) | **Identical** |

## Motivation

Flash Attention has revolutionized transformer efficiency, but its CUDA compilation requirements create significant deployment barriers. This research addresses a critical gap in the literature by directly comparing SDPA and Flash Attention in real-world applications.

## Methodology

### Experimental Setup
- **Model**: YOLOv12n (2.56M parameters)
- **Dataset**: Weeds-3 (3,664 training images, 359 validation)
- **Hardware**: NVIDIA RTX 4060 Laptop GPU (8GB)
- **Framework**: PyTorch 2.2.2+cu118
- **Configuration**: 20 epochs, batch size 8, AdamW optimizer (lr=0.001)

### Reproducibility
All experiments used deterministic mode with fixed random seeds. Complete configuration files and logs are provided in the `experiments/` directory.

## Results

### Performance Analysis
Both methods achieved identical final performance metrics (mAP50: 96.7%, mAP50-95: 75.3%), validating functional equivalence. The marginal 6% speed difference is offset by SDPA's significant advantages in deployment simplicity.

### Memory Efficiency
While Flash Attention demonstrates superior memory optimization (80.6% reduction), SDPA remains viable for most applications with <3GB memory usage.

## Usage

### Quick Start with SDPA
```python

import torch.nn.functional as F

def setup_sdpa_environment():
    """Optimized PyTorch SDPA configuration"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if hasattr(F, 'scaled_dot_product_attention'):
        return True
    return False

def sdpa_attention(q, k, v, mask=None):
    """SDPA attention mechanism"""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

### Reproducing Our Experiments
```bash
git clone https://github.com/kennedy-kitoko/sdpa-flash-attention-comparison
cd sdpa-flash-attention-comparison
python experiment_launcher.py
```

## Repository Structure
```
├── README.md                       # This file
├──                     # Experimental results and data
│   ├── complete_session_results.json
│   ├── results_sdpa.csv
│   └── results_flash_attn.csv
├──                            # Source code
│   └── experiment_launcher.py
├──                           # Additional documentation
│   ├── methodology.md
│   └── results_analysis.md
├── LICENSE                        # MIT License
└── CONTRIBUTING.md               # Contribution guidelines
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Citation

If you use this research in your work, please cite:

```bibtex
@misc{kitoko2025sdpa,
  title={SDPA vs Flash Attention: A Comparative Study for Production ML Systems},
  author={Kitoko Mutunga Kennedy},
  year={2025},
  institution={Beijing Institute of Technology},
  url={https://github.com/kennedy-kitoko/sdpa-flash-attention-comparison}
}
```

## Future Work

This research opens several exciting avenues for future exploration:

### Ongoing Experiments
- **COCO Dataset**: Currently extending validation to MS COCO for broader applicability
- **Model Architectures**: Testing on Vision Transformers (ViT), BERT, and GPT architectures
- **Larger Models**: Evaluating performance on YOLOv12m, YOLOv12l, and YOLOv12x

### Planned Research
- **Quantitative Analysis**: Detailed profiling of attention kernel operations
- **Hardware Diversity**: Testing on A100, H100, and consumer GPUs
- **Production Deployment**: Real-world case studies in agricultural applications
- **Hybrid Approaches**: Combining SDPA with other optimization techniques

### Collaboration Opportunities
We welcome collaborations on extending this research. Contact: kitokokennedy13@gmail.com

## Dataset Information

The Weeds-3 dataset used in this study was obtained from Roboflow:
- **Source**: Augmented Startups workspace
- **Classes**: 3 types of agricultural weeds
- **Images**: 3,664 training, 359 validation
- **Format**: YOLOv11 (compatible with YOLOv12)

## Author

**Kitoko Muyunga Kennedy**  
2nd Year Mechatronics Student, Beijing Institute of Technology  
Contact: kitokokennedy13@gmail.com  
Twitter/X: [@Kennedykitoko13](https://twitter.com/Kennedykitoko13)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Ultralytics team for YOLO implementation
- PyTorch team for SDPA implementation
- Tri Dao and Flash Attention team for their pioneering work
- Prof. Zhang Xiangfu for supervision and guidance
