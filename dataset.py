import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from config import (
    transform_only_input,
    transform_only_label,
    both_transform,
    transform_resize,
)


class MapDataset(Dataset):
    """Dataset for loading input, label, and mask images for inpainting training"""

    def __init__(self, root_dir, masked=False):
        """
        Args:
            root_dir: Root directory containing 'input', 'label', and 'mask' subdirectories
            masked: Whether to apply mask to input image
        """
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
