import torch
import glob
import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml


class DatasetSegmentation(Dataset):
    """
    For dataset structured as:
    dataset/train/2012_05/eventA_001.png + dataset/train/2012_05/masks/eventA_001.png
    (same for test/)
    """

    def __init__(self, config_file: dict, processor, mode: str):
        super().__init__()

        # Select the dataset path
        if mode == "train":
            base_path = config_file["DATASET"]["TRAIN_PATH"]
        else:
            base_path = config_file["DATASET"]["TEST_PATH"]

        self.processor = processor
        self.img_files = []
        self.mask_files = []

        # Traverse the dataset directories
        for root, dirs, files in os.walk(base_path):
            # Inspect image files in the current directory
            for file in files:
                if file.lower().endswith(".png"):
                    img_path = os.path.join(root, file)

                    # Skip mask directories and retain only input images
                    if "masks" in img_path:
                        continue

                    # Construct the corresponding mask path
                    mask_path = os.path.join(root, "masks", file)

                    if os.path.exists(mask_path):
                        self.img_files.append(img_path)
                        self.mask_files.append(mask_path)
                    else:
                        raise FileNotFoundError(f"⚠️ Mask not found for {img_path}")

        print(f"[{mode}] Found {len(self.img_files)} image-mask pairs in {base_path}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index: int):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        # Load the image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("1")  # Convert the mask to binary
        ground_truth_mask = np.array(mask)
        original_size = tuple(image.size)[::-1]  # (H, W)

        # Full-frame box prompt for SAM
        box = (0, 0, 1024, 1024)
        inputs = self.processor(image, original_size, box)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

        return inputs

def collate_fn(batch: torch.utils.data) -> list:
    """
    Used to get a list of dict as output when using a dataloader

    Arguments:
        batch: The batched dataset

    Return:
        (list): list of batched dataset so a list(dict)
    """
    return list(batch)