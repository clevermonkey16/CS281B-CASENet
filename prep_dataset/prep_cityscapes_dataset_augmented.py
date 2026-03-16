import os
import numpy as np
import time
import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
# Add project root to path so dataloader can be found when run from any cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from dataloader.cityscapes_data import CityscapesData

import config


class AddGaussianNoise:
    """Adds random Gaussian noise to a tensor. Sigma sampled uniformly per image."""
    def __init__(self, max_sigma=0.05):
        self.max_sigma = max_sigma

    def __call__(self, tensor):
        sigma = torch.rand(1).item() * self.max_sigma
        return tensor + torch.randn_like(tensor) * sigma


def get_dataloader(args, augmentation=False, random_erasing=False):
    # Define data files path.
    root_img_folder = "/workspace/CS281B-CASENet/cityscapes-preprocess/data_proc"
    root_label_folder = "/workspace/CS281B-CASENet/cityscapes-preprocess/data_proc"
    train_anno_txt = "/workspace/CS281B-CASENet/cityscapes-preprocess/data_proc/train.txt"
    val_anno_txt = "/workspace/CS281B-CASENet/cityscapes-preprocess/data_proc/val.txt"
    train_label_npy_dir = "/workspace/CS281B-CASENet/train_label_npy"
    val_label_npy_dir = "/workspace/CS281B-CASENet/val_label_npy"

    input_size = 472

    # ImageNet-standard preprocessing for MobileNetV3: RGB, [0,1], normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augmentation:
        # Enhanced geometric augmentations (synchronized between image and label)
        train_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0), ratio=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=PIL.Image.BILINEAR),
        ])
        train_label_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0), ratio=(0.75, 1.0), interpolation=PIL.Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=PIL.Image.NEAREST),
        ])

        # Image-only color/blur augmentations (applied before ToTensor)
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        gaussian_blur = transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3)

        # Post-tensor transforms: normalize, noise, optional erasing
        post_tensor = [normalize, AddGaussianNoise(max_sigma=0.05)]
        if random_erasing:
            post_tensor.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)))

        img_transform = transforms.Compose([
            train_augmentation,
            color_jitter,
            gaussian_blur,
            transforms.ToTensor(),
            *post_tensor,
        ])
    else:
        # Baseline geometric augmentations (same as original)
        train_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0), ratio=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])
        train_label_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0), ratio=(0.75, 1.0), interpolation=PIL.Image.NEAREST),
            transforms.RandomHorizontalFlip(),
        ])

        post_tensor = [normalize]
        if random_erasing:
            post_tensor.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)))

        img_transform = transforms.Compose([
            train_augmentation,
            transforms.ToTensor(),
            *post_tensor,
        ])

    train_dataset = CityscapesData(
        root_img_folder,
        root_label_folder,
        train_anno_txt,
        train_label_npy_dir,
        input_size,
        cls_num=args.cls_num,
        img_transform=img_transform,
        label_transform=transforms.Compose([
            transforms.ToPILImage(),
            train_label_augmentation,
            transforms.ToTensor(),
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    val_dataset = CityscapesData(
        root_img_folder,
        root_label_folder,
        val_anno_txt,
        val_label_npy_dir,
        input_size,
        cls_num=args.cls_num,
        img_transform=transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            normalize,
        ]),
        label_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([input_size, input_size], interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=max(1, args.batch_size//2), shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    return train_loader, val_loader

if __name__ == "__main__":
    args = config.get_args()
    args.batch_size = 1
    print("Testing with augmentation=True, random_erasing=True")
    train_loader, val_loader = get_dataloader(args, augmentation=True, random_erasing=True)
    for i, (img, target) in enumerate(train_loader):
        print("img.size():{0}".format(img.size()))
        print("target.size():{0}".format(target.size()))
        print("img min={0}, max={1}".format(img.min(), img.max()))
        break
