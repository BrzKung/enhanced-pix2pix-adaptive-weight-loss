from config_original import (
    DEVICE,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGE_SIZE,
    CHANNELS_IMG,
    L1_LAMBDA,
    ADAPTIVE_WEIGHT_BETA,
    USE_SOFTADAPT,
    SOFTADAPT_ITERATIONS,
    SOFTADAPT_ACCURACY_ORDER,
    LAMBDA_GP,
    NUM_EPOCHS,
    LOAD_MODEL,
    SAVE_MODEL,
    SAVE_MODEL_EPOCHS,
    CHECKPOINT,
    CHECKPOINT_DIR,
    LOAD_CHECKPOINT_GEN,
    LOAD_CHECKPOINT_DISC,
    LOAD_HISTORY_PATH,
    SAVE_HISTORY_VAL_DIR,
    both_transform,
    transform_only_input,
    transform_only_label,
    transform_only_mask,
    transform_resize,
)

from utils import save_checkpoint, load_checkpoint, load_lambda
from dataset import MapDataset
from models import Discriminator, Generator, test_disc, test_gen
from loss import PerceptualAndStyleLoss

# Training dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
from torcheval.metrics import FrechetInceptionDistance
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import torchvision.transforms as transforms

# torch.backends.cudnn.benchmark = True


def train_fn(
    epoch,
    disc,
    gen,
    loader,
    opt_disc,
    opt_gen,
    l1_loss,
    bce,
    g_scaler,
    d_scaler,
):
    loop = tqdm(loader, leave=True)
    sum_l1_loss = 0
    num_samples = 0

    global L1_LAMBDA
    global HISTORY

    for idx, (x, y, z) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32):
            y_fake = gen(x)  # random noise
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32):
            D_fake = disc(x, y_fake)

            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA

            sum_l1_loss += L1.item()
            num_samples += 1

            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
                L1_loss=L1.item(),
                L1_weight=L1_LAMBDA,
            )

    HISTORY["training_loss"].append(G_loss.item())
    HISTORY["epoch"].append(epoch)
    HISTORY["generator_loss"].append(G_fake_loss.item())
    HISTORY["discriminator_loss"].append(D_loss.item())
    HISTORY["l1_lambda"].append(L1_LAMBDA)
    HISTORY["l1_loss"].append(sum_l1_loss / num_samples)


def validate_fn(
    disc, gen, val_loader, opt_disc, opt_gen, perceptual_style_loss, l1_loss, bce
):
    global L1_LAMBDA
    total_L1, total_G_loss = 0, 0, 0, 0
    num_samples = 0

    print("Save model to Validate...")
    save_checkpoint(
        gen,
        opt_gen,
        filepath=f"{CHECKPOINT_DIR}/validate_gen.pth.tar",
    )
    save_checkpoint(
        disc,
        opt_disc,
        filepath=f"{CHECKPOINT_DIR}/validate_disc.pth.tar",
    )

    validate_disc = Discriminator(in_channels=3).to(DEVICE)
    validate_gen = Generator(in_channels=3, features=64).to(DEVICE)
    validate_opt_disc = optim.Adam(
        validate_disc.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    validate_opt_gen = optim.Adam(
        validate_gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )

    print("Load model to Validate...")
    load_checkpoint(
        f"{CHECKPOINT_DIR}/validate_gen.pth.tar",
        validate_gen,
        validate_opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        f"{CHECKPOINT_DIR}/validate_disc.pth.tar",
        validate_disc,
        validate_opt_disc,
        LEARNING_RATE,
    )

    real_list = []
    result_list = []
    resize = transforms.Resize(
        (299, 299), interpolation=transforms.InterpolationMode.BICUBIC
    )

    validate_gen.eval()
    with torch.no_grad():  # No gradients needed for validation
        for x, y, z in val_loader:
            # Cal val loss
            x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)
            y_fake = validate_gen(x)

            G_fake_loss = bce(
                validate_disc(x, y_fake), torch.ones_like(validate_disc(x, y_fake))
            )
            L1 = l1_loss(y_fake, y) * L1_LAMBDA

            G_loss = G_fake_loss + L1

            # Accumulate losses
            total_L1 += L1.item()
            total_G_loss += G_loss.item()

            num_samples += 1

            # Cal score
            y_fake = y_fake * 0.5 + 0.5  # remove normalization
            x = x * 0.5 + 0.5
            y = y * 0.5 + 0.5
            z = z.permute(0, 3, 1, 2)

            z_mask = (z > 0.5).to(dtype=torch.float32, device=DEVICE)
            inpainted = (
                (x * 2 - 1.0) * (1.0 - z_mask) + (y_fake * 2 - 1.0) * z_mask
            ) * 0.5 + 0.5

            result = resize(inpainted)
            real = resize(y)

            result_list.append(result)
            real_list.append(real)

    # gen.train()  # Switch back to training mode

    # Compute average losses
    avg_L1 = total_L1 / num_samples
    avg_G_loss = total_G_loss / num_samples

    print(
        f"Validation Loss -> G_loss: {avg_G_loss:.6f}, L1: {avg_L1:.6f}, L1_weight: {L1_LAMBDA}"
    )

    score, fid_score, psnr_score = calculateScore(result_list, real_list)

    return avg_G_loss, avg_L1, score, fid_score, psnr_score


def calculateScore(
    result_list,
    real_list,
):

    result_images = torch.cat(result_list, dim=0)
    real_images = torch.cat(real_list, dim=0)

    result_images = (result_images - result_images.min()) / (
        result_images.max() - result_images.min()
    )
    real_images = (real_images - real_images.min()) / (
        real_images.max() - real_images.min()
    )

    try:
        fid_score = calculateFID(real_images, result_images)
        psnr_score = calculatePSNR(real_images, result_images)

        score = 0.5 * psnr_score - 0.5 * fid_score
        score = score.item()
    except:
        score = -100

    return score, fid_score.item(), psnr_score.item()


def calculateFID(real_images, result_images):
    fid = FrechetInceptionDistance()
    fid.update(real_images, True)
    fid.update(result_images, False)
    fid_score = fid.compute()  # should < 10 -> 0
    return fid_score


def calculatePSNR(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr


def main():
    best_fid = 100
    steps_since_improvement = 0
    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if LOAD_MODEL:
        load_checkpoint(
            LOAD_CHECKPOINT_GEN,
            gen,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            LOAD_CHECKPOINT_DISC,
            disc,
            opt_disc,
            LEARNING_RATE,
        )
        load_lambda()

    train_dataset = MapDataset(root_dir=TRAIN_DIR, masked=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    val_dataset = MapDataset(root_dir=VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        train_fn(
            epoch + 1,
            disc,
            gen,
            train_loader,
            opt_disc,
            opt_gen,
            L1_LOSS,
            BCE,
            g_scaler,
            d_scaler,
        )

        val_loss, l1_loss, score, fid_score, psnr_score = validate_fn(
            disc,
            gen,
            val_loader,
            opt_disc,
            opt_gen,
            L1_LOSS,
            BCE,
        )

        print("FID Score : ", fid_score)

        if fid_score < best_fid:
            best_fid = fid_score
            print(f"New Best FID Score : {best_fid}")
            steps_since_improvement = 0
            save_checkpoint(
                gen,
                opt_gen,
                filepath=f"{CHECKPOINT_DIR}/best_gen_{CHECKPOINT + epoch + 1}.pth.tar",
            )
            save_checkpoint(
                disc,
                opt_disc,
                filepath=f"{CHECKPOINT_DIR}/best_disc_{CHECKPOINT + epoch + 1}.pth.tar",
            )
        else:
            steps_since_improvement += 1
            print("FID not improve : ", steps_since_improvement)

        HISTORY["validation_loss"].append(val_loss)
        HISTORY["fid_score"].append(fid_score)
        HISTORY["psnr_score"].append(psnr_score)

        if SAVE_MODEL and (epoch + 1) % SAVE_MODEL_EPOCHS == 0:
            save_checkpoint(
                gen,
                opt_gen,
                filepath=f"{CHECKPOINT_DIR}/gen_{CHECKPOINT + epoch + 1}.pth.tar",
            )
            save_checkpoint(
                disc,
                opt_disc,
                filepath=f"{CHECKPOINT_DIR}/disc_{CHECKPOINT + epoch + 1}.pth.tar",
            )

            history_df = pd.DataFrame(HISTORY)
            history_df.to_csv(
                f"{SAVE_HISTORY_VAL_DIR}/history_{CHECKPOINT + epoch + 1}.csv",
                index=False,
            )

        if steps_since_improvement > 500:
            print(f"⏹️ Early stopping: no improvement in 500 steps")
            break

    # save_checkpoint(gen, opt_gen, filepath=CHECKPOINT_GEN)
    # save_checkpoint(disc, opt_disc, filepath=CHECKPOINT_DISC)

    # save_some_examples(gen, val_loader, epoch, folder="evaluation")


# if __name__ == "__main__":
#     main()
HISTORY = {
    "epoch": [],
    "generator_loss": [],
    "discriminator_loss": [],
    "training_loss": [],
    "validation_loss": [],
    "l1_lambda": [],
    "l1_loss": [],
    "perc_lambda": [],
    "perc_loss": [],
    "style_lambda": [],
    "style_loss": [],
    "fid_score": [],
    "psnr_score": [],
}

main()
