"""Common utility functions for test scripts."""

import torch
import torchvision.utils as vutils
from torchvision.utils import save_image
import torchvision.transforms as transforms


def save_some_examples(gen, iter_val, folder, run_num, device):
    """
    Generate and save example outputs from generator.

    Args:
        gen: Generator model
        iter_val: Iterator over test data
        folder: Output folder path
        run_num: Run number for naming
        device: Device to run on (cuda/cpu)

    Returns:
        Tuple of (x, y, y_fake, z, inpainted)
    """
    x, y, z = next(iter_val)
    x, y, z = x.to(device), y.to(device), z.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        z = z.permute(0, 3, 1, 2)

        z_mask = (z > 0.5).to(dtype=torch.float32, device=device)
        inpainted = (
            (x * 2 - 1.0) * (1.0 - z_mask) + (y_fake * 2 - 1.0) * z_mask
        ) * 0.5 + 0.5

        # combine input, gen, label
        combined = torch.cat([x, y_fake, y, inpainted], dim=0)
        grid = vutils.make_grid(combined, nrow=4)
        # save_image(grid, folder + f"/result_{run_num}.png")
        save_image(inpainted, folder + f"/result_{run_num}.png")

    print("Run saved : ", run_num)
    # gen.train()
    return x, y, y_fake, z, inpainted


def save_some_examples_resized(gen, iter_val, folder, run_num, device, resize_size=512):
    """
    Generate and save example outputs from generator with resizing.

    Args:
        gen: Generator model
        iter_val: Iterator over test data
        folder: Output folder path
        run_num: Run number for naming
        device: Device to run on (cuda/cpu)
        resize_size: Size to resize output to

    Returns:
        Tuple of (x, y, y_fake, z, inpainted)
    """
    x, y, z = next(iter_val)
    x, y, z = x.to(device), y.to(device), z.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        z = z.permute(0, 3, 1, 2)

        z_mask = (z > 0.5).to(dtype=torch.float32, device=device)
        inpainted = (
            (x * 2 - 1.0) * (1.0 - z_mask) + (y_fake * 2 - 1.0) * z_mask
        ) * 0.5 + 0.5

        # Resize before saving
        resize = transforms.Resize((resize_size, resize_size))
        inpainted = resize(inpainted)
        save_image(inpainted, folder + f"/result_{run_num}.png")

    print("Run saved : ", run_num)
    return x, y, y_fake, z, inpainted
