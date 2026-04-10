import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "./datasets/train"
VAL_DIR = "./datasets/test"
TEST_DIR = "./datasets/test"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 0.1
ADAPTIVE_WEIGHT_BETA = 0.1
USE_SOFTADAPT = True
SOFTADAPT_ITERATIONS = 3
SOFTADAPT_ACCURACY_ORDER = None if SOFTADAPT_ITERATIONS <= 5 else 4
LAMBDA_GP = 10
NUM_EPOCHS = 1000
LOAD_MODEL = False
SAVE_MODEL = True
SAVE_MODEL_EPOCHS = 100
CHECKPOINT_DIR = "./checkpoints/pix2pix_original"
CHECKPOINT = 0
LOAD_CHECKPOINT_GEN = f"./checkpoints/pix2pix_original/gen_{CHECKPOINT}.pth.tar"
LOAD_CHECKPOINT_DISC = f"./checkpoints/pix2pix_original/disc_{CHECKPOINT}.pth.tar"
LOAD_HISTORY_PATH = f"./result/pix2pix_original/history_{CHECKPOINT}.csv"
SAVE_HISTORY_VAL_DIR = f"./result/pix2pix_original"

both_transform = A.Compose(
    [
        A.Resize(width=512, height=512),
    ],
    additional_targets={"image0": "image"},
    # [], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.2),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

transform_only_label = A.Compose(
    [
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        ToTensorV2(),
    ]
)

transform_resize = A.Resize(height=512, width=512)

# Utils
import torch


def save_checkpoint(model, optimizer, filepath="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint : {filepath}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(f"=> Loading checkpoint : {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_lambda():
    print("=> Loading lambda")
    global L1_LAMBDA

    lambda_df = pd.read_csv(LOAD_HISTORY_PATH)
    lasted_lamda = lambda_df.iloc[-1]

    L1_LAMBDA = lasted_lamda["l1_lambda"]

    print("l1 lambda: ", L1_LAMBDA)


# Datasets
import numpy as np

# import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir, masked=False):
        self.input_dir = root_dir + "/input"
        self.label_dir = root_dir + "/label"
        self.mask_dir = root_dir + "/mask"

        self.list_input_files = os.listdir(self.input_dir)
        self.list_label_files = os.listdir(self.label_dir)
        self.list_mask_files = os.listdir(self.mask_dir)

        self.list_input_files.sort()
        self.list_label_files.sort()
        self.list_mask_files.sort()

        self.masked = masked

    def __len__(self):
        return len(self.list_input_files)

    def __getitem__(self, index):
        input_img_file = self.list_input_files[index]
        input_img_path = os.path.join(self.input_dir, input_img_file)

        label_img_file = self.list_label_files[index]
        label_img_path = os.path.join(self.label_dir, label_img_file)

        mask_img_file = self.list_mask_files[index]
        mask_img_path = os.path.join(self.mask_dir, mask_img_file)

        input_image = np.array(Image.open(input_img_path).convert("RGB"))
        target_image = np.array(Image.open(label_img_path).convert("RGB"))
        mask_image = np.array(Image.open(mask_img_path).convert("RGB"))
        mask_image = (mask_image - mask_image.min()) / (
            mask_image.max() - mask_image.min()
        )

        input_image = input_image * (1.0 - mask_image) if self.masked else input_image
        # input_image = mask_image | input_image if self.masked else input_image

        mask_image_resized = transform_resize(image=mask_image)
        mask_image = mask_image_resized["image"]

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_label(image=target_image)["image"]

        return input_image, target_image, mask_image


# if __name__ == "__main__":
#     dataset = MapDataset(TRAIN_DIR)
#     loader = DataLoader(dataset, batch_size=5)
#     for x, y in loader:
#         print(x.shape)
#         save_image(x, "x.png")
#         save_image(y, "y.png")
#         import sys

#         sys.exit()

# Discriminator
import torch
import torch.nn as nn


# PatchGANs
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,  # concat input and label
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]

        # Apply stride=2 for the first 4 layers, then stride=1
        for idx, feature in enumerate(features[1:]):
            stride = 1 if idx == len(features[1:]) - 1 else 2
            layers.append(CNNBlock(in_channels, feature, stride))
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


def test_disc():
    x = torch.randn((5, 3, 512, 512))  # Input 512x512
    y = torch.randn((5, 3, 512, 512))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(preds.shape)  # Should print: torch.Size([5, 1, 30, 30])


# if __name__ == "__main__":
#     test()

# Generator
import torch
import torch.nn as nn


# U-Net
class Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, down=True, act="relu", use_dropout=False
    ):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            (
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    4,
                    2,
                    1,
                    bias=False,
                    padding_mode="reflect",
                )
                if down
                else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            ),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(
            features, features * 2, down=True, act="leaky", use_dropout=False
        )
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(
            features * 8, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(
            features * 2 * 2, features, down=False, act="relu", use_dropout=False
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, in_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test_gen():
    x = torch.randn((1, 3, 512, 512))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)


# if __name__ == "__main__":
#     test()

# Perceptual anb Style Loss
import torch
import torch.nn as nn
from torchvision import models, transforms


# Train
import torch
import pandas as pd

# from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim

# import config
# from dataset import MapDataset
# from generator_model import Generator
# from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
from torcheval.metrics import FrechetInceptionDistance
import torch.nn.functional as F
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
