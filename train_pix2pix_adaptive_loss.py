import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
CHECKPOINT_DIR = "./checkpoints/pix2pix_adaptive_loss"
CHECKPOINT = 0
LOAD_CHECKPOINT_GEN = (
    f"./checkpoints/pix2pix_adaptive_loss/recent/gen_{CHECKPOINT}.pth.tar"
)
LOAD_CHECKPOINT_DISC = (
    f"./checkpoints/pix2pix_adaptive_loss/recent/disc_{CHECKPOINT}.pth.tar"
)
HISTORY_PATH = f"./result/pix2pix_adaptive_loss/recent/history_{CHECKPOINT}.csv"

SAVE_HISTORY_VAL_DIR = "./result/pix2pix_adaptive_loss"

BEST_FID = float("inf")

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
    global PERC_LAMBDA
    global STYLE_LAMBDA

    lambda_df = pd.read_csv(HISTORY_PATH)
    lasted_lamda = lambda_df.iloc[-1]

    L1_LAMBDA = lasted_lamda["l1_lambda"]
    PERC_LAMBDA = lasted_lamda["perc_lambda"]
    STYLE_LAMBDA = lasted_lamda["style_lambda"]

    print("l1 lambda: ", L1_LAMBDA)
    print("perc lambda: ", PERC_LAMBDA)
    print("style lambda: ", STYLE_LAMBDA)


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

"""Definition of constants for odd finite difference (up to 5)"""
import numpy

# All constants are for forward finite difference method.
_FIRST_ORDER_COEFFICIENTS = numpy.array((-1, 1))
_THIRD_ORDER_COEFFICIENTS = numpy.array((-11/6, 3, -3/2, 1/3))
_FIFTH_ORDER_COEFFICIENTS = numpy.array((-137/60, 5, -5, 10/3, -5/4, 1/5))

"""Internal implementation of """
import numpy
from findiff import coefficients
# from ..constants._finite_difference_constants import (_FIRST_ORDER_COEFFICIENTS,_THIRD_ORDER_COEFFICIENTS, _FIFTH_ORDER_COEFFICIENTS)


def _get_finite_difference(input_array: numpy.array,
                           order: int = None,
                           verbose: bool = True):
    """Internal utility method for estimating rate of change.

    This function aims to approximate the rate of change for a loss function,
    which is used for the 'LossWeighted' and 'Normalized' variants of SoftAdapt.

    For even accuracy orders, we take advantage of the `findiff` package
    (https://findiff.readthedocs.io/en/latest/source/examples-basic.html).
    Accuracy orders of 1 (trivial), 3, and 5 are retrieved from an internal
    constants file. Due to the underlying mathematics of computing the
    coefficients, all accuracy orders higher than 5 must be an even number.

    Args:
        input_array: An array of floats containing loss evaluations at the
          previous 'n' points (as many points as the order) of the finite
          difference method.
        order: An integer indicating the order of the finite difference method
          we want to use. The function will use the length of the 'input_array'
          array if no values is provided.
        verbose: Whether we want the function to print out information about
          computations or not.

    Returns:
        A float which is the approximated rate of change between the loss
        points.

    Raises:
        ValueError: If the number of points in the `input_array` array is
          smaller than the order of accuracy we desire.
        Value Error: If the order of accuracy is higher than 5 and it is not an
          even number.
    """
    # First, we want to check the order and the number of loss points we are
    # given
    if order is None:
        order = len(input_array) - 1
        if verbose:
            print(f"==> Interpreting finite difference order as {order} since"
                  "no explicit order was specified.")
    else:
        if order > len(input_array):
            raise ValueError("The order of finite difference computations can"
                             "not be larger than the number of loss points. "
                             "Please check the order argument or wait until "
                             "enough points have been stored before calling the"
                             " method.")
        elif order + 1 < len(input_array):
            print(f"==> There are more points than 'order' + 1 ({order + 1}) "
                  f"points (array contains {len(input_array)} values). Function"
                  f"will use the last {order} elements of loss points for "
                  "computations.")
            input_array = input_array[(-1*order - 1):]

    order_is_even = order % 2 == 0
    # Next, we want to retrieve the correct coefficients based on the order
    if order > 5 and not order_is_even:
        raise ValueError("Accuracy orders larger than 5 must be even. Please "
                         "check the arguments passed to the function.")

    if order_is_even:
        constants = coefficients(deriv=1, acc=order)["forward"]["coefficients"]

    else:
        if order == 1:
            constants = _FIRST_ORDER_COEFFICIENTS
        elif order == 3:
            constants = _THIRD_ORDER_COEFFICIENTS
        else:
            constants = _FIFTH_ORDER_COEFFICIENTS

    pointwise_multiplication = [
        input_array[i] * constants[i] for i in range(len(constants))
    ]
    return numpy.sum(pointwise_multiplication)

_EPSILON = 1e-08

"""Implementaion of the base class for SoftAdapt."""

import torch
# from ..constants._stability_constants import _EPSILON
# from ..utilities._finite_difference import _get_finite_difference


class SoftAdaptBase():
    """Base model for any of the SoftAdapt variants.

    Attributes:
        epsilon: A float which is added to the denominator of a division for
          numerical stability.

    """

    def __init__(self):
        """Initializer of the base method."""
        self.epsilon = _EPSILON

    def _softmax(self,
                 input_tensor: torch.tensor,
                 beta: float = 1,
                 numerator_weights: torch.tensor = None,
                 shift_by_max_value: bool = True):
        """Implementation of SoftAdapt's modified softmax function.

        Args:
            input_tensor: A tensor of floats which will be used for computing
              the (modified) softmax function.
            beta: A float which is the scaling factor (as described in the
              manuscript).
            numerator_weights: A tensor of weights which are the actual value of
              of the loss components. This option is used for the
              "loss-weighted" variant of SoftAdapt.
            shift_by_max_value: A boolean indicating whether we want the values
              in the input tensor to be shifted by the maximum value.

        Returns:
            A tensor of floats that are the softmax results.

        Raises:
            None.

        """
        if shift_by_max_value:
            exp_of_input = torch.exp(beta * (input_tensor - input_tensor.max()))
        else:
            exp_of_input = torch.exp(beta * input_tensor)

        # This option will be used for the "loss-weighted" variant of SoftAdapt.
        if numerator_weights is not None:
            exp_of_input = torch.multiply(numerator_weights, exp_of_input)

        return exp_of_input / (torch.sum(exp_of_input) + self.epsilon)


    def _compute_rates_of_change(self,
                                 input_tensor:torch.tensor,
                                 order: int = 5,
                                 verbose: bool = True):
        """Base class method for computing loss functions rate of change.

        Args:
            input_tensor: A tensor of floats containing loss evaluations at the
              previous 'n' points (as many points as the order) of the finite
              difference method.
            order: An integer indicating the order of the finite difference
              method we want to use. The function will use the length of the
              'input_array' array if no values is provided.
            verbose: Whether we want the function to print out information about
              computations or not.

        Returns:
            The approximated derivative as a float value.

        Raises:
            None.

        """
        return _get_finite_difference(input_array = input_tensor.numpy(),
                                      order = order,
                                      verbose = verbose)
        
"""Implementaion of the loss-weighted variant of SoftAdapt."""

import torch
# from ..base._softadapt_base_class import SoftAdaptBase
from typing import Tuple


class LossWeightedSoftAdapt(SoftAdaptBase):
    """Class implementation of the loss-weighted SoftAdapt variant.

    The loss-weighted variant of SoftAdapt is described in section 3.1.1 of our
    manuscript (located at: https://arxiv.org/pdf/1912.12355.pdf).

    Attributes:
        beta: A float that is the 'beta' hyperparameter in our manuscript. If
          beta > 0, then softAdapt will pay more attention the worst performing
          loss component. If beta < 0, then SoftAdapt will assign higher weights
          to the better performing components. Beta==0 is the trivial case and
          all loss components will have coefficient 1.

        accuracy_order: An integer indicating the accuracy order of the finite
          volume approximation of each loss component's slope.
    """

    def __init__(self, beta: float = 0.1, accuracy_order: int = None):
        """SoftAdapt class initializer."""
        super().__init__()
        self.beta = beta
        # Passing "None" as the order of accuracy sets the highest possible
        # accuracy in the finite difference approximation.
        self.accuracy_order = accuracy_order

    def get_component_weights(self,
                               *loss_component_values: Tuple[torch.tensor],
                               verbose: bool = True):
        """Class method for SoftAdapt weights.

        Args:
            loss_component_values: A tuple consisting of the values of the each
              loss component that have been stored for the past 'n' iterations
              or epochs (as described in the manuscript).
            verbose: A boolean indicating user preference for whether internal
              functions should print out information and warning about
              computations.
        Returns:
            The computed weights for each loss components. For example, if there
            were 5 loss components, say (l_1, l_2, l_3, l_4, l_5), then the
            return tensor will be the weights (alpha_1, alpha_2, alpha_3,
            alpha_4, alpha_5) in the order of the loss components.

        Raises:
            None.

        """
        if len(loss_component_values) == 1:
            print("==> Warning: You have only passed on the values of one loss"
                  " component, which will result in trivial weighting.")

        rates_of_change = []
        average_loss_values = []

        for loss_points in loss_component_values:
            # Compute the rates of change for each one of the loss components.
            rates_of_change.append(
                self._compute_rates_of_change(loss_points,
                                              self.accuracy_order,
                                              verbose=verbose))
            average_loss_values.append(torch.mean(loss_points.float()))

        rates_of_change = torch.tensor(rates_of_change)
        average_loss_values = torch.tensor(average_loss_values)
        # Calculate the weight and return the values.
        return self._softmax(input_tensor=rates_of_change,
                             beta=self.beta,
                             numerator_weights = average_loss_values,
                             )

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
# from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt

from torcheval.metrics import FrechetInceptionDistance
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips

# torch.backends.cudnn.benchmark = True


def train_fn(
    epoch,
    disc,
    gen,
    loader,
    opt_disc,
    opt_gen,
    l1_loss,
    perceptual_style_loss,
    loss_weight_softadapt,
    bce,
    g_scaler,
    d_scaler,
    L1_losses,
    PERC_losses,
    STYLE_losses,
):
    loop = tqdm(loader, leave=True)
    sum_l1_loss = 0
    sum_perc_loss = 0
    sum_style_loss = 0
    num_samples = 0

    global L1_LAMBDA
    global PERC_LAMBDA
    global STYLE_LAMBDA
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
            PERC, STYLE = perceptual_style_loss(y_fake, y, y)

            PERC = PERC * PERC_LAMBDA
            STYLE = STYLE * STYLE_LAMBDA

            sum_l1_loss += L1.item()
            sum_perc_loss += PERC.item()
            sum_style_loss += STYLE.item()
            num_samples += 1

            G_loss = G_fake_loss + L1 + PERC + STYLE

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
                PERC_loss=PERC.item(),
                STYLE_loss=STYLE.item(),
                L1_weight=L1_LAMBDA,
                PERC_weight=PERC_LAMBDA,
                STYLE_weight=STYLE_LAMBDA,
            )

    L1_losses.append(sum_l1_loss / num_samples)
    PERC_losses.append(sum_perc_loss / num_samples)
    STYLE_losses.append(sum_style_loss / num_samples)

    HISTORY["epoch"].append(epoch)
    HISTORY["training_loss"].append(G_loss.item())
    HISTORY["generator_loss"].append(G_fake_loss.item())
    HISTORY["discriminator_loss"].append(D_loss.item())

    # with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32):
    # if USE_SOFTADAPT and epoch % SOFTADAPT_ITERATIONS == 0:

    # L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA = (
    #     loss_weight_softadapt.get_component_weights(
    #         torch.tensor(L1_losses),
    #         torch.tensor(PERC_losses),
    #         torch.tensor(STYLE_losses),
    #     )
    # )
    # L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA = (
    #     L1_LAMBDA.item(),
    #     PERC_LAMBDA.item(),
    #     STYLE_LAMBDA.item(),
    # )

    if USE_SOFTADAPT and epoch % SOFTADAPT_ITERATIONS == 0:
        L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA = loss_weight_softadapt.get_component_weights(
            torch.tensor(L1_losses),
            torch.tensor(PERC_losses),
            torch.tensor(STYLE_losses),
        )
        
        # L1_LAMBDA, PERC_LAMBDA, STYLE_LAMBDA = loss_weight_softadapt.update(
        #     [L1_losses[-1], PERC_losses[-1], STYLE_losses[-1]]
        # )

        print(
            f"Updated Weight Loss: L1_LAMBDA:{L1_LAMBDA}, PERC_LAMBDA: {PERC_LAMBDA}, STYLE_LAMBDA:{STYLE_LAMBDA}"
        )


def validate_fn(disc, gen, val_loader, l1_loss, perceptual_style_loss, bce):
    gen.eval()  # Set generator to evaluation mode

    global L1_LAMBDA
    global PERC_LAMBDA
    global STYLE_LAMBDA

    total_L1, total_PERC, total_STYLE, total_G_loss = 0, 0, 0, 0
    num_samples = 0

    real_list = []
    result_list = []

    resize = transforms.Resize(
        (299, 299), interpolation=transforms.InterpolationMode.BICUBIC
    )

    with torch.no_grad():  # No gradients needed for validation
        for x, y, z in val_loader:
            x, y, z = x.to(DEVICE), y.to(DEVICE), z.to(DEVICE)
            y_fake = gen(x)

            G_fake_loss = bce(disc(x, y_fake), torch.ones_like(disc(x, y_fake)))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            PERC, STYLE = perceptual_style_loss(y_fake, y, y)
            PERC = PERC * PERC_LAMBDA
            STYLE = STYLE * STYLE_LAMBDA

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

    gen.train()  # Switch back to training mode

    # Compute average losses
    avg_L1 = total_L1 / num_samples
    avg_PERC = total_PERC / num_samples
    avg_STYLE = total_STYLE / num_samples
    avg_G_loss = total_G_loss / num_samples

    print(
        f"Validation Loss -> G_loss: {avg_G_loss:.6f}, L1: {avg_L1:.6f}, L1_weight: {L1_LAMBDA}, PERC: {avg_PERC:.6f}, PERC_weight: {PERC_LAMBDA}, STYLE: {avg_STYLE:.6f}, STYLE_weight: {STYLE_LAMBDA}"
    )

    fid_score, psnr_score, ssim_score, lpips_score = calculateScore(
        result_list, real_list
    )

    HISTORY["validation_loss"].append(avg_G_loss)
    HISTORY["l1_lambda"].append(L1_LAMBDA)
    HISTORY["l1_loss"].append(avg_L1)  # คูณ lambda ไว้แล้ว
    HISTORY["perc_lambda"].append(PERC_LAMBDA)
    HISTORY["perc_loss"].append(avg_PERC)  # คูณ lambda ไว้แล้ว
    HISTORY["style_lambda"].append(STYLE_LAMBDA)
    HISTORY["style_loss"].append(avg_STYLE)
    HISTORY["fid_score"].append(fid_score)
    HISTORY["psnr_score"].append(psnr_score)
    HISTORY["ssim_score"].append(ssim_score)
    HISTORY["lpips_score"].append(lpips_score)

    return (
        avg_G_loss,
        avg_L1,
        avg_PERC,
        avg_STYLE,
        fid_score,
        psnr_score,
        ssim_score,
        lpips_score,
    )


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

    fid_score = calculateFID(real_images, result_images)
    psnr_score = calculatePSNR(real_images, result_images)
    ssim_score = calculateSSIM(real_images, result_images)
    lpips_score = calculateLPIPS(real_images, result_images)

    return (
        fid_score.item(),  # clip 20
        psnr_score.item(),
        ssim_score.item() * 10,  # clip 10
        lpips_score.item() * 100,  # clip 20
    )


def calculateFID(real_images, result_images):
    print("Calculating FID...")
    fid = FrechetInceptionDistance()
    fid.update(real_images, True)
    fid.update(result_images, False)
    fid_score = fid.compute()  # should < 10 -> 0
    return fid_score


def calculatePSNR(img1, img2, max_val=1.0):
    print("Calculating PSNR...")
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr


def calculateSSIM(real_images, result_images):
    print("Calculating SSIM...")
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    ssim_scores = ssim(real_images, result_images)
    return ssim_scores


def calculateLPIPS(real_images, result_images):
    print("Calculating LPIPS...")
    result_images = result_images * 2 - 1
    real_images = real_images * 2 - 1

    loss_fn = lpips.LPIPS(net="vgg").to(DEVICE)
    lpips_score = loss_fn(real_images, result_images).mean()
    return lpips_score


def main():
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

    perceptual_style_loss = PerceptualAndStyleLoss(
        style_layers=["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
        content_layers=["relu2_2", "relu3_3", "relu4_3", "relu5_1", "relu5_3"],
    ).to(DEVICE)

    # with torch.amp.autocast(device_type=DEVICE, dtype=torch.float32):
    # loss_weight_softadapt = SoftAdapt(num_losses=3, beta=ADAPTIVE_WEIGHT_BETA)
    # loss_weight_softadapt = SoftAdaptLossWeighting(num_losses=3, beta=ADAPTIVE_WEIGHT_BETA)
    
    loss_weight_softadapt = LossWeightedSoftAdapt(
        beta=ADAPTIVE_WEIGHT_BETA, accuracy_order=SOFTADAPT_ACCURACY_ORDER
    )
    # loss_weight_softadapt  = NormalizedSoftAdapt(beta=ADAPTIVE_WEIGHT_BETA, accuracy_order=SOFTADAPT_ACCURACY_ORDER)
    # loss_weight_softadapt = SoftAdapt(
    #    beta=ADAPTIVE_WEIGHT_BETA, accuracy_order=SOFTADAPT_ACCURACY_ORDER
    # )

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

    val_dataset = MapDataset(root_dir=VAL_DIR, masked=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    L1_losses = []
    PERC_losses = []
    STYLE_losses = []

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
            perceptual_style_loss,
            loss_weight_softadapt,
            BCE,
            g_scaler,
            d_scaler,
            L1_losses,
            PERC_losses,
            STYLE_losses,
        )

        (
            avg_G_loss,
            avg_L1,
            avg_PERC,
            avg_STYLE,
            fid_score,
            psnr_score,
            ssim_score,
            lpips_score,
        ) = validate_fn(disc, gen, val_loader, L1_LOSS, perceptual_style_loss, BCE)

        global BEST_FID

        if fid_score < BEST_FID:
            BEST_FID = fid_score
            print(f"New best FID: {BEST_FID}, saving model...")
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

        if (epoch + 1) % SOFTADAPT_ITERATIONS == 0:
            L1_losses = []
            PERC_losses = []
            STYLE_losses = []


HISTORY = {
    "epoch": [],
    "generator_loss": [],  # training
    "discriminator_loss": [],  # training
    "training_loss": [],  # training
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
}

main()
