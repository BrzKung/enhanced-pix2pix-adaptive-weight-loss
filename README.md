# Enhanced Pix2Pix with Adaptive Weight Loss

A pix2pix GAN implementation with adaptive loss weighting using reinforcement learning and Stable Baselines 3 for image inpainting tasks.

## Setup

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Deactivate Virtual Environment

```bash
deactivate
```

## Project Structure

```
enhanced-pix2pix-adaptive-weight-loss/
├── config.py                           # Stable Baseline variant config
├── config_original.py                  # Original GAN (L1 loss only)
├── config_fixed_loss.py               # Fixed L1+Perceptual+Style losses
├── config_adaptive_loss.py            # SoftAdapt adaptive weighting
├── dataset.py                         # Dataset loading utilities
├── models.py                          # Generator and Discriminator models
├── loss.py                            # Perceptual and Style loss functions
├── utils.py                           # Checkpoint and utility functions
├── env.py                             # RL environment for lambda tuning
├── train_pix2pix_original.py         # Original GAN training script
├── train_pix2pix_fixed_loss.py       # Fixed-weight multi-loss training
├── train_pix2pix_adaptive_loss.py    # SoftAdapt training
├── train_pix2pix_stable_baseline_3.py # RL-based lambda tuning
├── test_config.py                     # Test configuration
├── test_utils.py                      # Test utility functions
├── metrics.py                         # Evaluation metrics
├── test_pix2pix_original.py          # Test original variant
├── test_pix2pix_fixed_loss.py        # Test fixed-loss variant
├── test_pix2pix_adaptive_loss.py     # Test adaptive-loss variant
├── test_pix2pix_stable_baseline.py   # Test RL-tuned model
├── data/                              # Dataset storage
│   ├── train/
│   │   ├── input/                    # Masked input images
│   │   ├── label/                    # Clean target images
│   │   └── mask/                     # Inpainting masks
│   ├── val/
│   │   ├── input/
│   │   ├── label/
│   │   └── mask/
│   └── test/
│       ├── input/
│       ├── label/
│       └── mask/
├── checkpoints/                       # Model checkpoints
├── result/                            # Training results
└── readme.md                          # This file
```

## Data Preparation

### Data Structure

The project expects three subdirectories for each split (train/val/test):

- **input/**: Masked images (512×512 RGB) with missing regions
- **label/**: Ground truth complete images (512×512 RGB)
- **mask/**: Binary masks (512×512) indicating inpainting regions (white=255 for mask, black=0 for keep)

### Data Example

```
data/
├── train/
│   ├── input/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── label/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── mask/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
├── val/
│   ├── input/
│   ├── label/
│   └── mask/
└── test/
    ├── input/
    ├── label/
    └── mask/
```

### Preparing Your Data

1. **Organize images** into input, label, and mask folders
2. **Ensure consistent naming**: img_001.jpg, img_002.jpg, etc. (naming must match across all folders)
3. **Image size**: Resize all images to 512×512 pixels
4. **Masks**: Use white (255) for regions to inpaint, black (0) for regions to keep
5. **Split**: Typically 70% train, 15% val, 15% test

## Training Methods

### 1. Original Pix2Pix (L1 Loss Only)

Simple GAN training with L1 reconstruction loss:

```bash
python train_pix2pix_original.py
```

**Configuration** (config_original.py):
- Learning rate: 2e-4
- Batch size: 16
- Epochs: 1000
- Loss: L1 reconstruction loss only
- Use when: Quick prototyping, baseline comparison

**Output**: Checkpoints saved to `checkpoints/pix2pix_original/`

### 2. Fixed Multi-Loss Training

GAN with fixed-weight combination of L1, Perceptual, and Style losses:

```bash
python train_pix2pix_fixed_loss.py
```

**Configuration** (config_fixed_loss.py):
- Learning rate: 2e-4
- Batch size: 16
- Epochs: 5000
- Save interval: Every 1000 epochs
- Losses:
  - L1 Loss: λ₁ = 100
  - Perceptual Loss: λₚ = 10
  - Style Loss: λₛ = 100
- Use when: Want better perceptual quality with fixed weights

**Output**: Checkpoints saved to `checkpoints/pix2pix_fixed_loss/`

### 3. Adaptive Weighted Loss (SoftAdapt)

GAN with automatic loss weighting using SoftAdapt algorithm:

```bash
python train_pix2pix_adaptive_loss.py
```

**Configuration** (config_adaptive_loss.py):
- Learning rate: 2e-4
- Batch size: 16
- Epochs: 5000
- Adaptive weighting: Automatically scales λ₁, λₚ, λₛ based on loss magnitudes
- Use when: Want balanced gradients without manual tuning

**Output**: Checkpoints saved to `checkpoints/pix2pix_adaptive_loss/`

### 4. RL-Based Lambda Tuning (Stable Baselines 3)

Reinforcement learning agent that tunes loss weights to optimize FID score:

```bash
python train_pix2pix_stable_baseline_3.py
```

**Configuration** (config.py):
- RL Algorithm: PPO (Proximal Policy Optimization)
- Action space: [L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA]
- Observation space: Current lambdas + loss values + metrics
- Episode length: 5 steps (tuning iterations)
- Use when: Want optimal lambda combination discovered via RL

**Training Loop**:
1. Agent proposes lambda values
2. Train GAN for 1 epoch with those lambdas
3. Evaluate on validation set
4. Compute reward: -FID_score (lower FID = higher reward)
5. Agent learns which lambdas improve reward

**Output**: Best checkpoints saved to `checkpoints/pix2pix_stable_baseline/` with history CSV

**Training progress**:
```bash
# Monitor training logs
tail -f checkpoints/pix2pix_stable_baseline/recent/*.csv
```

## Testing

### Test Specific Variant

```bash
# Test original model
python test_pix2pix_original.py

# Test fixed-loss model
python test_pix2pix_fixed_loss.py

# Test adaptive-loss model
python test_pix2pix_adaptive_loss.py

# Test RL-optimized model
python test_pix2pix_stable_baseline.py
```

### Evaluation Metrics

Each test script computes:
- **FID Score**: Frechet Inception Distance (lower is better, target: < 10)
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better, target: > 20 dB)
- **SSIM**: Structural Similarity Index (higher is better, target: > 0.8)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)

### Generate Test Results

Results are saved to:
```
result/
├── pix2pix_original/
│   ├── recent/
│   │   └── result_1.png, result_2.png, ...
├── pix2pix_fixed_loss/
├── pix2pix_adaptive_loss/
└── pix2pix_stable_baseline/
```

## Configuration

### Modify Training Parameters

Edit the relevant `config_*.py` file:

```python
# config_fixed_loss.py
LEARNING_RATE = 2e-4      # Adjust optimizer learning rate
BATCH_SIZE = 16           # Increase for more GPU memory
NUM_EPOCHS = 5000         # Total training epochs
L1_LAMBDA = 100          # L1 loss weight
PERC_LAMBDA = 10         # Perceptual loss weight
STYLE_LAMBDA = 100       # Style loss weight
```

### Dataset Paths

Configure in each config file:
```python
TRAIN_DIR = "./datasets/train"
VAL_DIR = "./datasets/val"
CHECKPOINT_DIR = "./checkpoints/pix2pix_fixed_loss"
```

## Key Features

- **Multiple Training Variants**: Original, Fixed-weight, Adaptive, RL-optimized
- **Modular Architecture**: Separate modules for models, losses, datasets, utilities
- **Mixed Precision Training**: Uses torch.amp for faster training
- **Comprehensive Metrics**: FID, PSNR, SSIM, LPIPS evaluation
- **Checkpoint Management**: Save/load models and optimizer states
- **History Tracking**: CSV logs of all training metrics

## Dependencies

Key packages (see requirements.txt):
- torch, torchvision
- pytorch-cuda (or cpu version)
- stable-baselines3
- gymnasium
- albumentations
- lpips
- torcheval, torchmetrics
- tqdm, pandas

## Common Issues

**Issue**: CUDA out of memory
- **Solution**: Reduce `BATCH_SIZE` in config files

**Issue**: Training too slow
- **Solution**: Enable mixed precision (already enabled), or use smaller dataset

**Issue**: Poor inpainting quality
- **Solution**: Try fixed-loss or adaptive-loss variants, or run RL tuning for longer

**Issue**: Models not loading
- **Solution**: Check checkpoint paths match config settings

## References

- Original Pix2Pix: https://github.com/phillipi/pix2pix
- SoftAdapt: https://github.com/dr-aheydari/SoftAdapt
- Stable Baselines 3: https://stable-baselines3.readthedocs.io/
- Perceptual Losses: https://arxiv.org/abs/1603.08155
