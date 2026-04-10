import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

LEARNED_STEPS = 0
LEARN_STEPS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "./datasets/train"
VAL_DIR = "./datasets/test"
TEST_DIR = "./datasets/test"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
# L1_LAMBDA = 1
# PERC_LAMBDA = 1
# STYLE_LAMBDA = 1
ADAPTIVE_WEIGHT_BETA = 0.1
SOFTADAPT_ITERATIONS = 3
SOFTADAPT_ACCURACY_ORDER = None if SOFTADAPT_ITERATIONS <= 5 else 4
LAMBDA_GP = 10
NUM_EPOCHS = 5000  # not use
LOAD_MODEL = False
SAVE_MODEL = True
SAVE_MODEL_EPOCHS = 100
CHECKPOINT = 0

CHECKPOINT_DIR = "./checkpoints/pix2pix_stable_baseline"
CHECKPOINT_PATH = (
    f"./checkpoints/pix2pix_agent_stable_baseline/lambda_tuner_{LEARNED_STEPS}_steps"
)
LOAD_CHECKPOINT_GEN = f"./checkpoints/pix2pix_stable_baseline/gen_{CHECKPOINT}.pth.tar"
LOAD_CHECKPOINT_DISC = (
    f"./checkpoints/pix2pix_stable_baseline/disc_{CHECKPOINT}.pth.tar"
)

SAVE_HISTORY_TRAIN_DIR = f"./result/pix2pix_stable_baseline"
SAVE_HISTORY_VAL_DIR = f"./result/pix2pix_stable_baseline"

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


def getNormalized(list):
    eps = 1e-8
    mean = np.mean(list)
    std = np.std(list) if len(list) > 0 else 0
    z_scores = (list - mean) / (std + eps)

    return z_scores


def getDelta(previous_list, current):
    avg = np.mean(previous_list) if len(previous_list) > 0 else 0
    delta = current - avg

    return delta


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

        input_image = mask_image | input_image if self.masked else input_image

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
    print(preds.shape)  # ([1,1,30,30])


# if __name__ == "__main__":
#     test()

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


import torch
import torch.nn as nn
from torchvision import models, transforms


class PerceptualAndStyleLoss(nn.Module):
    def __init__(self, style_layers, content_layers, img_size=224):
        """
        Perceptual Loss using a pre-trained VGG19 model.
        Combines content loss and style loss.

        Args:
        - style_layers: List of VGG19 layer names (e.g., 'relu1_1') for style loss.
        - content_layers: List of VGG19 layer names (e.g., 'relu4_2') for content loss.
                          (Typically one layer for content loss).
        - style_weights: Dictionary mapping style layer names to their weights.
                         If None, equal weights of 1.0 will be used.
        - content_weight: Scalar weight for the content loss.
        - img_size: Target size to resize images before VGG processing.
        - use_l1: If True, uses L1 loss for feature/Gram matrix comparison;
                  otherwise, uses L2 (MSE) loss.
        """
        super(PerceptualAndStyleLoss, self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        # self.content_weight = content_weight
        self.img_size = img_size
        self.criterion = nn.MSELoss()

        # VGG16
        self.layer_names_map = {
            "2": "relu1_2",
            "7": "relu2_2",
            "14": "relu3_3",
            "19": "relu4_2",
            "21": "relu4_3",
            "24": "relu5_1",
            "28": "relu5_3",
        }

        # Define layer mappings for VGG19 (sequential index to common name)
        # self.layer_names_map = {
        #     "0": "relu1_1",
        #     "5": "relu2_1",
        #     "10": "relu3_1",
        #     "19": "relu4_1",
        #     "28": "relu5_1",  # Style layers
        #     "21": "relu4_2",  # Common content layer
        #     # Add more if you need other specific VGG19 layers
        # }

        # Load pre-trained VGG19 features and set up feature extraction
        self.vgg = models.vgg16(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)  # Freeze VGG parameters

        # Determine the maximum layer index we need to go up to
        # to efficiently stop feature extraction
        self.max_layer_index = -1
        for idx_str, name in self.layer_names_map.items():
            if name in style_layers or name in content_layers:
                self.max_layer_index = max(self.max_layer_index, int(idx_str))

        # Trim VGG to only include layers up to the highest required one
        # This makes forward pass more efficient if only early layers are needed
        self.feature_extractor = nn.Sequential(
            *list(self.vgg.children())[: self.max_layer_index + 1]
        )

        # Default style weights if not provided
        # self.style_weights = style_weights

        # Preprocessing transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def _get_features(self, image_tensor):
        """
        Extracts features from specified VGG layers.
        Input image_tensor is expected to be [B, C, H, W] but NOT normalized yet.
        """
        # Ensure image is in the correct range for VGG (0-1, then normalize)
        # Assuming input is already scaled [0,1] or [0,255] and needs normalization
        # If your input image is already normalized, skip this step.
        # For typical neural networks, input images should be [0,1] or [0,255] and then normalized.
        # Here we assume the input to this _get_features is raw image data (e.g., pixel values before normalization).
        # We will apply the full transform including normalization.

        # Apply the full transform (resize and normalize)
        # Batch processing: apply transform to each image in batch if batch size > 1
        # print('input = ' , image_tensor.min(), image_tensor.max())
        processed_image = (
            torch.stack([self.transform(img) for img in image_tensor.squeeze(1)])
            if image_tensor.dim() == 5
            else self.transform(image_tensor)
        )
        # print('normalized = ',processed_image.min(), processed_image.max())

        features = {}
        x = processed_image
        for name, layer in self.feature_extractor._modules.items():
            x = layer(x)
            # Check if the current layer's index is one we need
            if name in self.layer_names_map:
                layer_name_str = self.layer_names_map[name]
                if (
                    layer_name_str in self.style_layers
                    or layer_name_str in self.content_layers
                ):
                    features[layer_name_str] = x
        return features

    # def gram_matrix(self, input_tensor):
    #     """Compute the Gram matrix for style representation."""
    #     B, C, H, W = input_tensor.size()
    #     features = input_tensor.view(B, C, H * W) # Reshape to (B, C, N) where N = H*W
    #     gram = torch.bmm(features, features.transpose(1, 2)) # Batch matrix-matrix product (B, C, C)
    #     return gram / (C * H * W) # Normalize
    def gram_matrix(self, input):
        a, b, c, d = (
            input.size()
        )  # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f-map (N=c*d)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

    def forward(self, generated_image, target_content_image, target_style_image):
        """
        Computes the perceptual loss.

        Args:
        - generated_image: The output image from your generator network. (Batch, C, H, W)
                           Expected to be in the same range as target images before VGG processing.
        - target_content_image: The ground truth content image. (Batch, C, H, W)
        - target_style_image: The reference style image. (Batch, C, H, W)

        Returns:
        - total_perceptual_loss: The combined content and style loss.
        """

        # generated_image = generated_image / torch.norm(generated_image, dim=1, keepdim=True)
        # target_content_image = target_content_image / torch.norm(target_content_image, dim=1, keepdim=True)
        # target_style_image = target_style_image / torch.norm(target_style_image, dim=1, keepdim=True)

        # Extract features for all images
        gen_features = self._get_features(generated_image)
        content_features = self._get_features(target_content_image)
        style_features = self._get_features(target_style_image)

        # Calculate Content Loss
        content_loss = 0.0
        for layer_name in self.content_layers:
            # We use detach() for target features to prevent gradients from flowing into target
            content_loss += self.criterion(
                gen_features[layer_name], content_features[layer_name].detach()
            )

        # Calculate Style Loss
        style_loss = 0.0
        for layer_name in self.style_layers:
            gen_gram = self.gram_matrix(gen_features[layer_name])
            target_gram = self.gram_matrix(
                style_features[layer_name].detach()
            )  # Detach target Grams

            # Handle potential inf/NaN (though normalization should prevent this)
            # gen_gram = torch.nan_to_num(gen_gram, nan=0.0, posinf=0.0, neginf=0.0)
            # target_gram = torch.nan_to_num(target_gram, nan=0.0, posinf=0.0, neginf=0.0)

            style_loss += self.criterion(gen_gram, target_gram)

        # total_loss = content_loss + style_loss

        return [content_loss, style_loss]


import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
import itertools
import torchvision.transforms as transforms
from torcheval.metrics import FrechetInceptionDistance
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips


class LambdaTuningEnv(gym.Env):
    def __init__(self):
        super(LambdaTuningEnv, self).__init__()

        # Action space: L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA
        self.action_space = spaces.Box(
            low=np.array([-100.0, -100.0, -100.0]),
            high=np.array([100.0, 100.0, 100.0]),
            dtype=np.float64,
        )

        # Observation space (you can design this differently if needed)
        self.observation_space = spaces.Box(
            low=np.array([-100.0] * 8), high=np.array([100.0] * 8), dtype=np.float64
        )

        # self.best_val_loss = 100
        self.best_image_score = None

        self.current_episode = 1
        self.current_step = 1
        self.overall_step = 1
        self.best_score = None

        self.current_l1_lambda = 0
        self.current_perc_lambda = 0
        self.current_style_lambda = 0
        self.current_l1_loss = 0
        self.current_perc_loss = 0
        self.current_style_loss = 0
        self.current_val_loss = 0

        self.disc = Discriminator(in_channels=3).to(DEVICE)
        self.gen = Generator(in_channels=3, features=64).to(DEVICE)
        self.opt_disc = optim.Adam(
            self.disc.parameters(),
            lr=LEARNING_RATE,
            betas=(0.5, 0.999),
        )
        self.opt_gen = optim.Adam(
            self.gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
        )
        self.g_scaler = torch.amp.GradScaler()
        self.d_scaler = torch.amp.GradScaler()

        if LOAD_MODEL:
            load_checkpoint(
                LOAD_CHECKPOINT_GEN,
                self.gen,
                self.opt_gen,
                LEARNING_RATE,
            )
            load_checkpoint(
                LOAD_CHECKPOINT_DISC,
                self.disc,
                self.opt_disc,
                LEARNING_RATE,
            )

        self.HISTORY_VAL = {
            "step": [],
            "overall_step": [],
            "episode": [],
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
            "ssim_score": [],
            "lpips_score": [],
            "reward": [],
        }

        # self.HISTORY_TRAIN = {
        #     "epoch": [],
        #     "total_loss": [],
        #     "generator_loss": [],
        #     "discriminator_loss": [],
        #     #    "l1_total_loss": [],
        #     "l1_lambda": [],
        #     "l1_loss": [],
        #     #    "perc_total_loss": [],
        #     "perc_lambda": [],
        #     "perc_loss": [],
        #     #    "style_total_loss": [],
        #     "style_lambda": [],
        #     "style_loss": [],
        # }

        self.BCE = nn.BCEWithLogitsLoss()
        self.L1_LOSS = nn.L1Loss()
        # VGG16
        self.PERC_STYLE_LOSS = PerceptualAndStyleLoss(
            style_layers=["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
            content_layers=["relu2_2", "relu3_3", "relu4_3", "relu5_1", "relu5_3"],
            # content_layers=["relu4_2"],
        ).to(DEVICE)

        # #VGG19
        # self.PERC_STYLE_LOSS = PerceptualAndStyleLoss(
        #     style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
        #     content_layers=["relu4_2"],
        # ).to(DEVICE)

        # self.PERC_LOSS = PerceptualLoss(layers=[0, 1, 2, 3]).to(DEVICE)
        # self.STYLE_LOSS = StyleLoss(layers=[2, 7, 14]).to(DEVICE)

    def reset(self, seed=None, options=None):
        # Reset your model or training stats here
        # self.last_lambdas = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.best_score = None
        self.current_step = 1
        print("reset !!!")

        return np.array([1.0] * 3 + [100] * 5), {}

    def step(self, action):
        L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA = action

        print("==========================")
        print("Episode: ", self.current_episode)
        print("Step: ", self.current_step)
        print("overall_step: ", self.overall_step)
        print(f"L1_LAMBDA: {L1_LAMBDA}")
        print(f"PERC_LAMBDA: {PERC_LAMBDA}")
        print(f"STYLE_LAMBDA: {STYLE_LAMBDA}")

        # Inject these into your training loop
        (
            val_loss,
            l1_loss,
            perc_loss,
            style_loss,
            L1_LAMBDA,
            PERC_LAMBDA,
            STYLE_LAMBDA,
            score,  # 0.5 psnr - 0.5 fid
            fid_score,
            psnr_score,
            ssim_score,
            lpips_score,
        ) = self.train_and_evaluate(L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA)

        done = False
        image_score = -fid_score  #####
        print("Image Score : ", image_score)

        reward = image_score  #####
        print(f"reward: {reward}")

        self.HISTORY_VAL["reward"].append(reward)
        self.HISTORY_VAL["step"].append(self.current_step)
        self.HISTORY_VAL["overall_step"].append(self.overall_step)
        self.HISTORY_VAL["episode"].append(self.current_episode)
        self.HISTORY_VAL["validation_loss"].append(val_loss)
        self.HISTORY_VAL["l1_loss"].append(l1_loss)
        self.HISTORY_VAL["l1_lambda"].append(L1_LAMBDA)
        self.HISTORY_VAL["perc_loss"].append(perc_loss)
        self.HISTORY_VAL["perc_lambda"].append(PERC_LAMBDA)
        self.HISTORY_VAL["style_loss"].append(style_loss)
        self.HISTORY_VAL["style_lambda"].append(STYLE_LAMBDA)
        self.HISTORY_VAL["fid_score"].append(fid_score)
        self.HISTORY_VAL["psnr_score"].append(psnr_score)
        self.HISTORY_VAL["ssim_score"].append(ssim_score)
        self.HISTORY_VAL["lpips_score"].append(lpips_score)

        if self.best_image_score is None or reward > self.best_image_score:
            self.best_image_score = reward
            print(f"New Best Image Score : {self.best_image_score}")
            save_checkpoint(
                self.gen,
                self.opt_gen,
                filepath=f"{CHECKPOINT_DIR}/best_gen_{CHECKPOINT + self.overall_step}.pth.tar",
            )
            save_checkpoint(
                self.disc,
                self.opt_disc,
                filepath=f"{CHECKPOINT_DIR}/best_disc_{CHECKPOINT + self.overall_step}.pth.tar",
            )

        # Save lambdas if they are the best so far (optional)
        if (self.current_step >= 5) or (
            self.best_score is not None and self.best_score < reward
        ):
            done = True

            print(f"Episode: {self.current_episode} is done...")
            self.current_episode += 1
            self.best_score = reward

        if self.best_score is None:
            # self.current_episode += 1
            self.best_score = reward

        # reward = np.clip(reward, a_min=-10, a_max=10).item()
        # print(f"clipped reward: {reward}")

        if SAVE_MODEL and (self.overall_step) % SAVE_MODEL_EPOCHS == 0:
            save_checkpoint(
                self.gen,
                self.opt_gen,
                filepath=f"{CHECKPOINT_DIR}/gen_{CHECKPOINT + self.overall_step}.pth.tar",
            )
            save_checkpoint(
                self.disc,
                self.opt_disc,
                filepath=f"{CHECKPOINT_DIR}/disc_{CHECKPOINT + self.overall_step}.pth.tar",
            )

            history_val_df = pd.DataFrame(self.HISTORY_VAL)
            history_val_df.to_csv(
                f"{SAVE_HISTORY_VAL_DIR}/history_{CHECKPOINT + self.overall_step}.csv",
                index=False,
            )

            # history_train_df = pd.DataFrame(self.HISTORY_TRAIN)
            # history_train_df.to_csv(
            #     f"{SAVE_HISTORY_TRAIN_DIR}/history_{CHECKPOINT + self.overall_step}.csv",
            #     index=False,
            # )

        self.current_step += 1
        self.overall_step += 1

        return (
            np.array(
                [
                    L1_LAMBDA,
                    PERC_LAMBDA,
                    STYLE_LAMBDA,
                    l1_loss,
                    perc_loss,
                    style_loss,
                    fid_score,
                    lpips_score,
                ]
            ),
            reward,
            done,
            False,
            {"best_image_score": self.best_image_score},
        )

    def train_and_evaluate(self, L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA):
        # ==== Example setup, replace with your actual model/training logic ====
        NUM_EPOCHS = 1  # Keep it short for fast evaluation per action

        # Setup: model, optimizer, loader, etc. (or use self.model, etc. if passed in __init__)
        # Here we assume these objects already exist, or are initialized globally
        # global disc
        # global gen
        # global opt_disc
        # global opt_gen

        # disc = Discriminator(in_channels=3).to(DEVICE)
        # gen = Generator(in_channels=3, features=64).to(DEVICE)
        # opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
        # opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        # BCE = nn.BCEWithLogitsLoss()
        # L1_LOSS = nn.L1Loss()
        # PERC_LOSS = PerceptualLoss(layers=[0, 1, 2, 3]).to(DEVICE)
        # STYLE_LOSS = StyleLoss(layers=[2, 7, 14]).to(DEVICE)

        # load_checkpoint(
        #     LOAD_CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE,
        # )
        # load_checkpoint(
        #     LOAD_CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
        # )

        train_dataset = MapDataset(root_dir=TRAIN_DIR, masked=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

        # global g_scaler
        # global d_scaler

        # g_scaler = torch.amp.GradScaler()
        # d_scaler = torch.amp.GradScaler()

        val_dataset = MapDataset(root_dir=VAL_DIR, masked=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        L1_losses = []
        PERC_losses = []
        STYLE_losses = []

        # Run a short training loop
        for epoch in range(NUM_EPOCHS):
            self.train_fn(
                epoch + 1,
                # self.disc,
                # self.gen,
                train_loader,
                # self.opt_disc,
                # self.opt_gen,
                # self.L1_LOSS,
                # self.PERC_STYLE_LOSS,
                # self.BCE,
                # self.g_scaler,
                # self.d_scaler,
                L1_losses,
                PERC_losses,
                STYLE_losses,
                L1_LAMBDA,
                PERC_LAMBDA,
                STYLE_LAMBDA,
            )

            # if self.overall_step % SAVE_MODEL_EPOCHS == 0:
            (
                val_loss,
                l1_loss,
                perc_loss,
                style_loss,
                score,
                fid_score,
                psnr_score,
                ssim_score,
                lpips_score,
            ) = self.validate_fn(
                # self.disc,
                # self.gen,
                val_loader,
                # self.L1_LOSS,
                # self.PERC_STYLE_LOSS,
                # self.BCE,
                L1_LAMBDA,
                PERC_LAMBDA,
                STYLE_LAMBDA,
            )

        # score = self.calculateScore(val_loader)
        # score = 0

        return (
            val_loss,
            l1_loss,
            perc_loss,
            style_loss,
            L1_LAMBDA,
            PERC_LAMBDA,
            STYLE_LAMBDA,
            score,
            fid_score,
            psnr_score,
            ssim_score,
            lpips_score,
        )

    def train_fn(
        self,
        epoch,
        # disc,
        # gen,
        loader,
        # opt_disc,
        # opt_gen,
        # l1_loss,
        # perceptual_style_loss,
        # bce,
        # g_scaler,
        # d_scaler,
        L1_losses,
        PERC_losses,
        STYLE_losses,
        L1_LAMBDA,
        PERC_LAMBDA,
        STYLE_LAMBDA,
    ):
        loop = tqdm(loader, leave=True)
        sum_l1_loss = 0
        sum_perc_loss = 0
        sum_style_loss = 0
        num_samples = 0

        L1_LAMBDA_CLIPED, PERC_LAMBDA_CLIPED, STYLE_LAMBDA_CLIPED = np.clip(
            [L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA], a_min=0.0001, a_max=100
        )

        for idx, (x, y, z) in enumerate(loop):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Train Discriminator
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32):
                y_fake = self.gen(x)  # random noise

                D_real = self.disc(x, y)
                D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
                D_fake = self.disc(x, y_fake.detach())
                D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = D_real_loss + D_fake_loss

            self.opt_disc.zero_grad()
            self.d_scaler.scale(D_loss).backward()
            self.d_scaler.step(self.opt_disc)
            self.d_scaler.update()

            # Train generator
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32):
                D_fake = self.disc(x, y_fake)

                G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
                L1 = self.L1_LOSS(y_fake, y) * L1_LAMBDA_CLIPED
                PERC, STYLE = self.PERC_STYLE_LOSS(y_fake, y, y)
                PERC = PERC * PERC_LAMBDA_CLIPED
                STYLE = STYLE * STYLE_LAMBDA_CLIPED
                # PERC = perceptual_loss(y_fake, y) * PERC_LAMBDA
                # STYLE = style_loss(y_fake, y) * STYLE_LAMBDA

                sum_l1_loss += L1.item()
                sum_perc_loss += PERC.item()
                sum_style_loss += STYLE.item()
                num_samples += 1

                G_loss = G_fake_loss + L1 + PERC + STYLE

            self.opt_gen.zero_grad()
            self.g_scaler.scale(G_loss).backward()
            self.g_scaler.step(self.opt_gen)
            self.g_scaler.update()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                    D_loss=D_loss.item(),
                    G_loss=G_loss.item(),
                    L1_loss=L1.item(),
                    PERC_loss=PERC.item(),
                    STYLE_loss=STYLE.item(),
                    L1_weight=L1_LAMBDA,
                    PERC_weight=PERC_LAMBDA,
                    STYLE_weight=STYLE_LAMBDA,
                )

        L1_losses.append(sum_l1_loss / num_samples)
        PERC_losses.append(sum_perc_loss / num_samples)
        STYLE_losses.append(sum_style_loss / num_samples)

        self.HISTORY_VAL["generator_loss"].append(G_fake_loss.item())
        self.HISTORY_VAL["discriminator_loss"].append(D_loss.item())
        self.HISTORY_VAL["training_loss"].append(G_loss.item())

        # self.HISTORY_TRAIN["total_loss"].append(G_loss.item())
        # self.HISTORY_TRAIN["epoch"].append(self.overall_step)
        # self.HISTORY_TRAIN["generator_loss"].append(G_fake_loss.item())
        # self.HISTORY_TRAIN["discriminator_loss"].append(D_loss.item())
        # # self.HISTORY_TRAIN["l1_total_loss"].append(sum_l1_loss/num_samples) # คูณ lambda แล้ว
        # self.HISTORY_TRAIN["l1_lambda"].append(L1_LAMBDA)
        # self.HISTORY_TRAIN["l1_loss"].append(sum_l1_loss / num_samples)
        # # self.HISTORY_TRAIN["perc_total_loss"].append(sum_perc_loss/num_samples) # คูณ lambda แล้ว
        # self.HISTORY_TRAIN["perc_lambda"].append(PERC_LAMBDA)
        # self.HISTORY_TRAIN["perc_loss"].append(sum_perc_loss / num_samples)
        # # self.HISTORY_TRAIN["style_total_loss"].append(sum_style_loss/num_samples) # คูณ lambda แล้ว
        # self.HISTORY_TRAIN["style_lambda"].append(STYLE_LAMBDA)
        # self.HISTORY_TRAIN["style_loss"].append(sum_style_loss / num_samples)

    def validate_fn(
        self,
        # disc,
        # gen,
        val_loader,
        # l1_loss,
        # perceptual_style_loss,
        # bce,
        L1_LAMBDA,
        PERC_LAMBDA,
        STYLE_LAMBDA,
    ):
        # self.gen.eval()  # Set generator to evaluation mode
        total_L1, total_PERC, total_STYLE, total_G_loss = 0, 0, 0, 0
        num_samples = 0

        print("Save model to Validate...")
        save_checkpoint(
            self.gen,
            self.opt_gen,
            filepath=f"{CHECKPOINT_DIR}/validate_gen.pth.tar",
        )
        save_checkpoint(
            self.disc,
            self.opt_disc,
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

        L1_LAMBDA_CLIPED, PERC_LAMBDA_CLIPED, STYLE_LAMBDA_CLIPED = np.clip(
            [L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA], a_min=0.0001, a_max=100
        )

        validate_gen.eval()
        with torch.no_grad():  # No gradients needed for validation
            for x, y, z in val_loader:
                # Cal val loss
                x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)
                y_fake = validate_gen(x)

                G_fake_loss = self.BCE(
                    validate_disc(x, y_fake), torch.ones_like(validate_disc(x, y_fake))
                )
                L1 = self.L1_LOSS(y_fake, y) * L1_LAMBDA_CLIPED
                PERC, STYLE = self.PERC_STYLE_LOSS(y_fake, y, y)
                PERC = PERC * PERC_LAMBDA_CLIPED
                STYLE = STYLE * STYLE_LAMBDA_CLIPED
                # PERC = perceptual_loss(y_fake, y) * PERC_LAMBDA
                # STYLE = style_loss(y_fake, y) * STYLE_LAMBDA

                G_loss = G_fake_loss + L1 + PERC + STYLE

                # Accumulate losses
                total_L1 += L1.item()
                total_PERC += PERC.item()
                total_STYLE += STYLE.item()
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

        # self.gen.train()  # Switch back to training mode

        # Compute average losses
        avg_L1 = total_L1 / num_samples
        avg_PERC = total_PERC / num_samples
        avg_STYLE = total_STYLE / num_samples
        avg_G_loss = total_G_loss / num_samples

        print(
            f"Validation Loss -> G_loss: {avg_G_loss:.6f}, L1: {avg_L1:.6f}, L1_weight: {L1_LAMBDA}, PERC: {avg_PERC:.6f}, PERC_weight: {PERC_LAMBDA}, STYLE: {avg_STYLE:.6f}, STYLE_weight: {STYLE_LAMBDA}"
        )

        score, fid_score, psnr_score, ssim_score, lpips_score = self.calculateScore(
            result_list, real_list
        )

        return (
            avg_G_loss,
            avg_L1,
            avg_PERC,
            avg_STYLE,
            score,
            fid_score,
            psnr_score,
            ssim_score,
            lpips_score,
        )

    def calculateScore(
        self,
        # gen,
        # val_loader,
        result_list,
        real_list,
    ):
        # self.gen.eval()  # Set generator to evaluation mode

        # real_list = []
        # result_list = []
        # resize = transforms.Resize(
        #     (299, 299), interpolation=transforms.InterpolationMode.BICUBIC
        # )

        # with torch.no_grad():  # No gradients needed for validation
        #     for x, y, z in val_loader:
        #         x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)
        #         y_fake = self.gen(x)

        #         y_fake = y_fake * 0.5 + 0.5  # remove normalization
        #         x = x * 0.5 + 0.5
        #         y = y * 0.5 + 0.5
        #         z = z.permute(0, 3, 1, 2)

        #         z_mask = (z > 0.5).to(dtype=torch.float32, device=DEVICE)
        #         inpainted = (
        #             (x * 2 - 1.0) * (1.0 - z_mask) + (y_fake * 2 - 1.0) * z_mask
        #         ) * 0.5 + 0.5

        #         result = resize(inpainted)
        #         real = resize(y)

        #         result_list.append(result)
        #         real_list.append(real)

        result_images = torch.cat(result_list, dim=0)
        real_images = torch.cat(real_list, dim=0)

        result_images = (result_images - result_images.min()) / (
            result_images.max() - result_images.min()
        )
        real_images = (real_images - real_images.min()) / (
            real_images.max() - real_images.min()
        )

        try:
            fid_score = self.calculateFID(real_images, result_images)
            psnr_score = self.calculatePSNR(real_images, result_images)
            ssim_score = self.calculateSSIM(real_images, result_images)
            lpips_score = self.calculateLPIPS(real_images, result_images)

            score = 0.5 * psnr_score - 0.5 * fid_score
            score = score.item()
        except:
            score = -100

        return (
            score,
            fid_score.item(),  # clip 20
            psnr_score.item(),
            ssim_score.item() * 10,  # clip 10
            lpips_score.item() * 100,  # clip 20
        )

    def calculateFID(self, real_images, result_images):
        print("Calculating FID...")
        fid = FrechetInceptionDistance()
        fid.update(real_images, True)
        fid.update(result_images, False)
        fid_score = fid.compute()  # should < 10 -> 0
        return fid_score

    def calculatePSNR(self, img1, img2, max_val=1.0):
        print("Calculating PSNR...")
        mse = F.mse_loss(img1, img2)
        psnr = 10 * torch.log10(max_val**2 / mse)
        return psnr

    def calculateSSIM(self, real_images, result_images):
        print("Calculating SSIM...")
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        ssim_scores = ssim(real_images, result_images)
        return ssim_scores

    def calculateLPIPS(self, real_images, result_images):
        print("Calculating LPIPS...")
        result_images = result_images * 2 - 1
        real_images = real_images * 2 - 1

        loss_fn = lpips.LPIPS(net="vgg").to(DEVICE)
        lpips_score = loss_fn(real_images, result_images).mean()
        return lpips_score


from stable_baselines3.common.callbacks import BaseCallback


class EarlyStopCallback(BaseCallback):
    def __init__(self, patience=500, verbose=0):
        super().__init__(verbose)
        self.patience = patience
        self.best_mean_fid = float("inf")
        self.steps_since_improvement = 0

    def _on_step(self) -> bool:
        if self.locals.get("infos") is not None:
            image_score = [
                info.get("best_image_score", 100) for info in self.locals["infos"]
            ]

            if len(image_score) > 0:
                mean_image_score = sum(image_score) / len(image_score)
                if mean_image_score < self.best_mean_fid:
                    self.best_mean_image_score = mean_image_score
                    self.steps_since_improvement = 0
                else:
                    self.steps_since_improvement += 1
                    print("Best Image Score : ", self.best_mean_image_score)
                    print("Image Score not improve : ", self.steps_since_improvement)

        if self.steps_since_improvement >= self.patience:
            print(f"⏹️ Early stopping: no improvement in {self.patience} steps")
            return False  # stop training
        return True


from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np


class VecNormalizeHistory(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.obs_history = []
        self.reward_history = []

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # store history
        self.obs_history.append(obs.copy())
        self.reward_history.append(rewards.copy())

        return obs, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        self.obs_history = []
        self.reward_history = []
        return obs


from stable_baselines3 import PPO  # or any other RL algorithm
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv([LambdaTuningEnv])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=20.0)  #####
env = VecNormalizeHistory(env)

if LOAD_MODEL:
    print(f"load model at {CHECKPOINT_PATH}")
    model = PPO.load(
        CHECKPOINT_PATH,
        env=env,
        verbose=1,
        tensorboard_log="./logs",
    )
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs",
        n_steps=16,
        n_epochs=4,
        use_sde=True,
        sde_sample_freq=4,
        ent_coef=0,
    )

# eval_callback = EvalCallback(
#     env,
#     best_model_save_path="./checkpoints/",
#     # log_path="./logs/",
#     eval_freq=64,
#     deterministic=True,
#     render=False,
# )

checkpoint_callback = CheckpointCallback(
    save_freq=64, save_path="./checkpoints/", name_prefix="lambda_tuner"
)

early_stopping_callback = EarlyStopCallback(patience=500)

callback = CallbackList([checkpoint_callback])

model.learn(total_timesteps=5000, reset_num_timesteps=True, callback=callback)

import numpy as np
import pandas as pd


def save_vec_normalize_history(array_list, filename="output.csv", columns=[]):
    # flatten each (1,8) array to (8,)
    flat = [arr.reshape(-1) for arr in array_list]

    # stack into shape (n_rows, 8)
    stacked = np.vstack(flat)

    # create column names
    num_cols = stacked.shape[1]

    # create dataframe
    df = pd.DataFrame(stacked, columns=columns)

    # save to csv
    df.to_csv(filename, index=False)

    print(f"Saved {len(df)} rows with {num_cols} columns to {filename}")


save_vec_normalize_history(
    env.obs_history,
    filename=f"{SAVE_HISTORY_VAL_DIR}/obs_normalize_history.csv",
    columns=[
        "l1_lambda",
        "perc_lambda",
        "style_lambda",
        "l1_loss",
        "perc_loss",
        "style_loss",
        "fid_score",
        "lpips_score",
    ],
)

save_vec_normalize_history(
    env.reward_history,
    filename=f"{SAVE_HISTORY_VAL_DIR}/reward_normalize_history.csv",
    columns=[
        "reward",
    ],
)

# model.save(f'./checkpoints/lambda_tuner_{LEARNED_STEPS}_steps')

# history_df = pd.DataFrame(HISTORY)
# history_df.to_csv(f"./result/stable_baseline_512/history_{LEARNED_STEPS + LEARN_STEPS}.csv", index=False)
