"""Evaluation metrics for model testing."""

import torch
import torch.nn.functional as F
from torcheval.metrics import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips


def compute_psnr(img1, img2, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum pixel value (default 1.0 for normalized images)

    Returns:
        PSNR score in dB
    """
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr


def compute_fid(real_images, result_images):
    """
    Compute Fréchet Inception Distance (FID).

    Args:
        real_images: Tensor of real images (normalized 0-1)
        result_images: Tensor of generated images (normalized 0-1)

    Returns:
        FID score (lower is better)
    """
    # Normalize to 0-1 range
    real_images = (real_images - real_images.min()) / (
        real_images.max() - real_images.min()
    )
    result_images = (result_images - result_images.min()) / (
        result_images.max() - result_images.min()
    )

    fid = FrechetInceptionDistance()
    fid.update(real_images, True)
    fid.update(result_images, False)
    fid_score = fid.compute()
    return fid_score


def compute_ssim(real_images, result_images, device="cpu"):
    """
    Compute Structural Similarity Index Measure (SSIM).

    Args:
        real_images: Tensor of real images
        result_images: Tensor of generated images
        device: Device to run on (cuda/cpu)

    Returns:
        SSIM score (higher is better, max 1.0)
    """
    # Convert to grayscale
    result_gray = torch.mean(result_images, dim=1, keepdim=True)
    real_gray = torch.mean(real_images, dim=1, keepdim=True)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_scores = ssim(real_gray, result_gray)
    return ssim_scores


def compute_lpips(real_images, result_images, device="cpu"):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        real_images: Tensor of real images (normalized to [-1, 1])
        result_images: Tensor of generated images (normalized to [-1, 1])
        device: Device to run on (cuda/cpu)

    Returns:
        LPIPS score (lower is better)
    """
    # Normalize to [-1, 1]
    result_images = result_images * 2 - 1
    real_images = real_images * 2 - 1

    loss_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_score = loss_fn(real_images, result_images).mean()
    return lpips_score


def evaluate_all_metrics(real_images, result_images, device="cpu"):
    """
    Compute all evaluation metrics.

    Args:
        real_images: Tensor of real images
        result_images: Tensor of generated images
        device: Device to run on (cuda/cpu)

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "psnr": compute_psnr(real_images, result_images).item(),
        "fid": compute_fid(real_images, result_images).item(),
        "ssim": compute_ssim(real_images, result_images, device).item(),
        "lpips": compute_lpips(real_images, result_images, device).item(),
    }

    return metrics


def print_metrics(metrics):
    """Print all metrics in a readable format."""
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"PSNR Score:  {metrics['psnr']:.4f} dB (higher is better)")
    print(f"FID Score:   {metrics['fid']:.4f} (lower is better)")
    print(f"SSIM Score:  {metrics['ssim']:.4f} (higher is better, max 1.0)")
    print(f"LPIPS Score: {metrics['lpips']:.4f} (lower is better)")
    print("=" * 50 + "\n")
