import torch
import os
import numpy as np
from PIL import Image

checkpoint = 4849
file_path = f"./result/pix2pix_fixed_loss/recent/{checkpoint}_epochs"
checkpoint_path = f"./checkpoints/pix2pix_fixed_loss/recent/gen_{checkpoint}.pth.tar"
TEST_DIR = f"./datasets/test"
history_file_name = f"/history_{checkpoint}.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

        input_image = mask_image | input_image if self.masked else input_image

        mask_image_resized = transform_resize(image=mask_image)
        mask_image = mask_image_resized["image"]

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_label(image=target_image)["image"]

        return input_image, target_image, mask_image


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


import torch

# import config
from torchvision.utils import save_image
import torchvision.utils as vutils


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(f"=> Loading checkpoint : {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_some_examples(gen, iter_val, folder, run_num):
    x, y, z = next(iter_val)
    x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        x = x * 0.5 + 0.5
        y = y * 0.5 + 0.5
        z = z.permute(0, 3, 1, 2)

        z_mask = (z > 0.5).to(dtype=torch.float32, device=DEVICE)
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


import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

test_dataset = MapDataset(root_dir=TEST_DIR, masked=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_gen_model = Generator(in_channels=3, features=64).to(DEVICE)
test_opt_gen = optim.Adam(
    test_gen_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)
load_checkpoint(f"{checkpoint_path}", test_gen_model, test_opt_gen, LEARNING_RATE)

import torchvision.transforms as transforms

testing_num = 1
iter_test_loader = iter(test_loader)
resize = transforms.Resize(
    (299, 299), interpolation=transforms.InterpolationMode.BICUBIC
)

real_list = []
result_list = []

while True:
    try:
        x, y, y_fake, mask, inpainted = save_some_examples(
            test_gen_model, iter_test_loader, folder=f"{file_path}", run_num=testing_num
        )

        result = resize(inpainted)
        real = resize(y)

        # result = (inpainted)
        # real = (y)

        result_list.append(result)
        real_list.append(real)

        testing_num += 1
    except StopIteration:
        break

import lpips

result_images = torch.cat(result_list, dim=0)
real_images = torch.cat(real_list, dim=0)

result_images = result_images * 2 - 1
real_images = real_images * 2 - 1

loss_fn = lpips.LPIPS(net="vgg").to(DEVICE)
lpips_score = loss_fn(real_images, result_images).mean()
print("lpips score : ", lpips_score)


from torcheval.metrics import FrechetInceptionDistance

result_images = torch.cat(result_list, dim=0)
real_images = torch.cat(real_list, dim=0)

result_images = (result_images - result_images.min()) / (
    result_images.max() - result_images.min()
)
real_images = (real_images - real_images.min()) / (
    real_images.max() - real_images.min()
)


fid = FrechetInceptionDistance()
fid.update(real_images, True)
fid.update(result_images, False)
fid_score = fid.compute()  # should < 10 -> 0
print("fid score : ", fid_score)

from torchmetrics.image import StructuralSimilarityIndexMeasure

result_images = torch.cat(result_list, dim=0)
real_images = torch.cat(real_list, dim=0)

result_gray = torch.mean(result_images, dim=1, keepdim=True)
real_gray = torch.mean(real_images, dim=1, keepdim=True)

ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
ssim_scores = ssim(real_gray, result_gray)
print("ssim score : ", ssim_scores)  # should > 0.8 -> 1

import torch
import torch.nn.functional as F

result_images = torch.cat(result_list, dim=0)
real_images = torch.cat(real_list, dim=0)


def compute_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr


psnr_score = compute_psnr(real_images, result_images)
print("prnr score : ", psnr_score)  # -> เยอะๆ
