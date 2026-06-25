# HSI and 2D Natural Image Super-Resolution Framework

This repository hosts a comprehensive suite of deep-learning tools for **Hyperspectral Image (HSI) Super-Resolution (3D)** and **Natural Image Super-Resolution (2D)**. Developed under the SVNIT-ISRO project, this framework includes 6 specialized 3D CNN architectures for HSI spatial-spectral enhancement and a generalized 2D upscaling CLI tool powered by Real-ESRGAN.

---

## 🛰️ Key Features

### 1. Hyperspectral Image (3D) Super-Resolution
HSIs capture information across hundreds of spectral bands, creating a spatial-spectral 3D cube. This framework processes HSI patches as 3D tensors `(Bands, Height, Width)` and uses **3D Convolutions** to jointly extract spatial and spectral features without distortion.
* **6 Model Architectures**:
  - `HSISuperRes3D`: Baseline 3D CNN utilizing trilinear upsampling.
  - `D3CNN`: Stack of 3D residual blocks for global residual learning.
  - `Lightweight3DCNN`: Fast, 3-layer architecture optimized for rapid inference.
  - `Deep3DResNet`: A deep 3D residual network (8 blocks) for high-fidelity spatial/spectral reconstruction.
  - `Dense3DNet`: A 3D dense network that propagates multi-level features via skip connections.
  - `Hybrid3DNet`: Runs parallel spatial `(1, 3, 3)` and spectral `(3, 1, 1)` pathways fused for maximum correlation.
* **Loss Functions**:
  - Combined Mean Squared Error (MSE) spatial loss and **Spectral Angle Mapper (SAM) loss** in radians to prevent spectral distortion.
* **Evaluation Metrics**: average PSNR, SSIM, and SAM in degrees.

### 2. 2D Natural Image Super-Resolution
An automated command-line utility for upscaling natural RGB images (e.g. JPG, PNG) using **Real-ESRGAN** and **SRGAN/SRResNet**.
* Supports **2x, 4x, and 8x** upscaling factors.
* Features **automatic spatial tiling** to prevent GPU VRAM Out-of-Memory (OOM) on large source images.
* Supports **batch directory processing** or **single-file** upscaling.
* GPU-accelerated with automatic half-precision (FP16) configuration.

---

## 📂 Directory Structure

```
.
├── implement/
│   ├── model/
│   │   ├── dataset/         # Pavia, PaviaU, and Indian Pines .mat datasets (git-ignored)
│   │   ├── weights/         # Pre-trained Real-ESRGAN models (git-ignored)
│   │   ├── checkpoints/     # Saved training checkpoints and metrics history (git-ignored)
│   │   ├── output of img incRes/ # Upscaled natural image outputs (git-ignored)
│   │   ├── model.ipynb      # HSI Model training & interactive Gradio UI notebook
│   │   ├── train_hsi.py     # Standalone HSI Model training CLI script
│   │   ├── upscale_image.py # Standalone 2D image upscaling CLI script
│   │   └── upscale_guitars.py # Batch upscaling utility script
│   ├── super-resolution/   # 2D SRGAN & SRResNet codebase (VGG features & adversarial loss)
│   └── .venv/               # Virtual environment directory (git-ignored)
├── info/                    # References and dataset download links
└── paper/                   # Key academic reference papers on HSI / Mamba SR
```

---

## 🛠️ CLI Tool Guides

### 1. Training 3D HSI Models (`train_hsi.py`)
Run the standalone training script on HSI datasets (like Indian Pines or Pavia University) with full parameter control.

**Example Command:**
```powershell
python implement/model/train_hsi.py --model Deep3DResNet --dataset implement/model/dataset/Indian_pines_corrected.mat --epochs 50 --batch-size 4 --lr 1e-3 --scale 2
```

**CLI Parameters:**
* `--model`: Select model architecture (`HSISuperRes3D`, `D3CNN`, `Lightweight3DCNN`, `Deep3DResNet`, `Dense3DNet`, `Hybrid3DNet`).
* `--dataset`: Path to the `.mat` dataset file (must contain `indian_pines_corrected` or `paviaU`).
* `--epochs`: Number of training epochs (default: `50`).
* `--batch-size`: Batch size (default: `4`).
* `--lr`: Learning rate (default: `1e-3`).
* `--scale`: Upscaling factor (2 or 4, default: `2`).
* `--patience`: Early stopping patience (default: `10`).
* `--save-dir`: Directory to save models (default: `checkpoints`).

**Outputs**:
* Best model saved as `checkpoints/<ModelName>_best.pth` containing model state, optimizer state, and validation metrics.
* Complete training metrics logged to `checkpoints/<ModelName>_history.json`.

---

### 2. Upscaling 2D Images (`upscale_image.py`)
Enhance natural images or a folder of images by 2x, 4x, or 8x.

**Example Command (Single Image):**
```powershell
python implement/model/upscale_image.py --input implement/super-resolution/img/lenna.png --output output_dir/upscaled_lenna.png --scale 4
```

**Example Command (Batch Directory):**
```powershell
python implement/model/upscale_image.py --input input_folder/ --output output_folder/ --scale 4 --format png
```

**CLI Parameters:**
* `--input`, `-i`: Path to input image file or directory containing images.
* `--output`, `-o`: Path to output image file or directory to save results.
* `--scale`, `-s`: Upscaling scale factor (`2`, `4`, or `8`, default: `4`).
* `--tile`: Tile size for spatial division (default: `400`). Lower value fits smaller VRAM.
* `--format`: Output image format (`png`, `jpg`, `webp`, default: `png`).
* `--weights-dir`: Directory containing the pre-trained weights (default: `weights`).

---

## ⚡ Setup & Dependencies

1. Ensure Python 3.11 is installed.
2. Activate the project's virtual environment:
   ```powershell
   # Windows
   .\implement\.venv\Scripts\Activate.ps1
   ```
3. Install base requirements (from `implement/super-resolution/requirements.txt` or standard PyTorch requirements):
   * `torch` (CUDA compatible)
   * `torchvision`
   * `numpy`
   * `scipy`
   * `scikit-image`
   * `realesrgan`
   * `basicsr`
   * `argparse`

---
