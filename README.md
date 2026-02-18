# Video Super Resolution

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/video-super-resolution/blob/main/video_super_resolution.ipynb)

Classical upsampling vs. SRCNN deep learning for frame-by-frame video super resolution.

## Overview

Builds a complete video super resolution pipeline covering:
- **Data pipeline** — LR/HR pair generation, patch extraction, train/val/test splits
- **Classical baselines** — nearest neighbour, bilinear, bicubic, Lanczos
- **SRCNN** — Super Resolution CNN implemented in PyTorch (Dong et al., 2014)
- **Quantitative evaluation** — PSNR & SSIM comparison across all methods
- **Video processing** — frame-by-frame SR inference, HR video reconstruction

## Architecture — SRCNN

```
LR (bicubic upsampled)
    → Conv(9×9, 64) + ReLU    # patch extraction
    → Conv(1×1, 32) + ReLU    # non-linear mapping
    → Conv(5×5, 1)             # reconstruction
    → SR output
```

Only **~20K parameters** — lightweight and fast to train and deploy.

## Results

| Method      | PSNR (dB) ↑ | SSIM ↑ |
|-------------|-------------|--------|
| Nearest     | ~24         | ~0.70  |
| Bilinear    | ~26         | ~0.78  |
| Bicubic     | ~27         | ~0.80  |
| Lanczos     | ~27         | ~0.81  |
| **SRCNN**   | **~29+**    | **~0.85+** |

*Results for ×3 upscaling. Vary by scale factor and training epochs.*

## Requirements

```
pip install torch torchvision numpy scipy matplotlib scikit-image opencv-python Pillow
```

## Usage

```bash
jupyter notebook video_super_resolution.ipynb
```

Or open in Colab via the badge above — no local setup needed.
Enable GPU: `Runtime → Change runtime type → T4 GPU`

## Using Your Own Video

In Section 6, uncomment Option B:
```python
SYNTHETIC_VIDEO = 'your_video.mp4'
```

## Files

| File | Description |
|---|---|
| `video_super_resolution.ipynb` | Main notebook |
| `srcnn_best.pth` | Best model weights (generated after training) |
| `srcnn_weights.pth` | Final model weights (generated after training) |
| `sr_output.avi` | Super resolved video output (generated after running) |
| `README.md` | This file |

## References

- Dong, C. et al. (2014). *Learning a Deep Convolutional Network for Image Super-Resolution.* ECCV 2014. [arXiv:1501.00092](https://arxiv.org/abs/1501.00092)
