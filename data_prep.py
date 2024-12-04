# data_prep.py

import os
import random
import numpy as np
from tqdm import tqdm
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import cv2

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths
dataset_path = "./custom_mnist_yolo"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")

os.makedirs(images_path + "/train", exist_ok=True)
os.makedirs(images_path + "/val", exist_ok=True)
os.makedirs(labels_path + "/train", exist_ok=True)
os.makedirs(labels_path + "/val", exist_ok=True)

transform = transforms.ToTensor()
mnist_train = MNIST(root="./data", train=True, download=True, transform=transform)
mnist_val = MNIST(root="./data", train=False, download=True, transform=transform)


def create_images(dataset, split):
    for idx in tqdm(range(len(dataset)), desc=f"Creating {split} data"):
        img, label = dataset[idx]
        img_np = img.numpy()[0]
        digit_size = img_np.shape[0]

        canvas_size = 620
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

        max_offset = canvas_size - digit_size
        x_min = random.randint(0, max_offset)
        y_min = random.randint(0, max_offset)
        x_max = x_min + digit_size
        y_max = y_min + digit_size

        canvas[y_min:y_max, x_min:x_max] = (img_np * 255).astype(np.uint8)

        image_filename = f"{idx}.png"
        cv2.imwrite(os.path.join(images_path, split, image_filename), canvas)

        x_center = (x_min + x_max) / 2 / canvas_size
        y_center = (y_min + y_max) / 2 / canvas_size
        width = (x_max - x_min) / canvas_size
        height = (y_max - y_min) / canvas_size

        label_filename = f"{idx}.txt"
        with open(os.path.join(labels_path, split, label_filename), "w") as f:
            f.write(f"{label} {x_center} {y_center} {width} {height}\n")


create_images(mnist_train, "train")
create_images(mnist_val, "val")
