import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "./datasets/train"
VAL_DIR = "./datasets/test"
TEST_DIR = "./datasets/test"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 0.1
PERC_LAMBDA = 0.1
STYLE_LAMBDA = 0.1
ADAPTIVE_WEIGHT_BETA = 0.1
USE_SOFTADAPT = True
SOFTADAPT_ITERATIONS = 3
SOFTADAPT_ACCURACY_ORDER = None if SOFTADAPT_ITERATIONS <= 5 else 4
LAMBDA_GP = 10
NUM_EPOCHS = 5000
LOAD_MODEL = False
SAVE_MODEL = True
SAVE_MODEL_EPOCHS = 1000
CHECKPOINT = 0

CHECKPOINT_DIR = "./checkpoints/pix2pix_fixed_loss"
LOAD_CHECKPOINT_GEN = (
    f"./checkpoints/pix2pix_fixed_loss/recent/gen_{CHECKPOINT}.pth.tar"
)
LOAD_CHECKPOINT_DISC = (
    f"./checkpoints/pix2pix_fixed_loss/recent/disc_{CHECKPOINT}.pth.tar"
)
HISTORY_PATH = f"./result/pix2pix_fixed_loss/recent/history_{CHECKPOINT}.csv"
SAVE_HISTORY_VAL_DIR = "./result/pix2pix_fixed_loss"

BEST_FID = float("inf")

# Data augmentation transforms
both_transform = A.Compose(
    [
        A.Resize(width=512, height=512),
    ],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
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
