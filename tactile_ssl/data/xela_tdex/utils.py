from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.stats import multivariate_normal

import random

from copy import deepcopy as copy

# Alexnet means and stds
TACTILE_IMAGE_MEANS = [0.485, 0.456, 0.406]
TACTILE_IMAGE_STDS = [0.229, 0.224, 0.225]

# These constants are used to clamp
TACTILE_PLAY_DATA_CLAMP_MIN = -500
TACTILE_PLAY_DATA_CLAMP_MAX = 500

XELA_IMG_ORDER = {
    "3aftc_palm_link": [4, 0],
    "link_15_4x4_palm_link": [10, 0],
    "link_14_4x4_palm_link": [14, 0],
    "0aftc_palm_link": [0, 6],
    "link_2_4x4_palm_link": [6, 6],
    "link_1A_4x4_palm_link": [10, 6],
    "link_1B_4x4_palm_link": [14, 6],
    "1aftc_palm_link": [0, 12],
    "link_6_4x4_palm_link": [6, 12],
    "link_5A_4x4_palm_link": [10, 12],
    "link_5B_4x4_palm_link": [14, 12],
    "2aftc_palm_link": [0, 18],
    "link_10_4x4_palm_link": [6, 18],
    "link_9A_4x4_palm_link": [10, 18],
    "link_9B_4x4_palm_link": [14, 18],
    "ahr_palm_2_4x6_palm_link": [18, 6],
    "ahr_palm_1_4x6_palm_link": [18, 12],
    "ahr_palm_3_4x6_palm_link": [22, 12],
}


def get_tactile_augmentations(img_size: Tuple[int, int], scale: int = 1):
    tactile_aug = T.Compose(
        [
            T.RandomApply(
                nn.ModuleList([T.RandomResizedCrop(img_size, scale=(0.9, 1), antialias=True)]),
                p=0.5,
            ),
            T.RandomApply(nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.5),
            T.Normalize(mean=TACTILE_IMAGE_MEANS * scale, std=TACTILE_IMAGE_STDS * scale),
        ]
    )
    return tactile_aug


def tactile_scale_transform(image):
    image = (image - TACTILE_PLAY_DATA_CLAMP_MIN) / (TACTILE_PLAY_DATA_CLAMP_MAX - TACTILE_PLAY_DATA_CLAMP_MIN)
    return image


def tactile_clamp_transform(image):
    image = torch.clamp(image, min=TACTILE_PLAY_DATA_CLAMP_MIN, max=TACTILE_PLAY_DATA_CLAMP_MAX)
    return image


class TactileImage:
    def __init__(self, tactile_image_size=224, shuffle_type=None):
        self.shuffle_type = shuffle_type
        self.size = tactile_image_size

        self.transform = T.Compose(
            [
                T.Resize((tactile_image_size, tactile_image_size), antialias=True),
                T.Lambda(tactile_clamp_transform),
                T.Lambda(tactile_scale_transform),
            ]
        )

    def get(self, type, tactile_values):
        if type == "whole_hand":
            return self.get_whole_hand_tactile_image(tactile_values)
        if type == "single_sensor":
            return self.get_single_tactile_image(tactile_values)
        if type == "stacked":
            return self.get_stacked_tactile_image(tactile_values)

    def get_stacked_tactile_image(self, tactile_values):
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = tactile_image.view(15, 4, 4, 3)  # Just making sure that everything stays the same
        tactile_image = torch.permute(tactile_image, (0, 3, 1, 2))
        tactile_image = tactile_image.reshape(-1, 4, 4)
        return self.transform(tactile_image)

    def get_single_tactile_image(self, tactile_value):
        tactile_image = torch.FloatTensor(tactile_value)  # tactile_value.shape: (16,3)
        tactile_image = tactile_image.view(4, 4, 3)
        tactile_image = torch.permute(tactile_image, (2, 0, 1))
        return self.transform(tactile_image)

    def get_whole_hand_tactile_image(self, tactile_values):
        tactile_image = np.zeros((26, 24, 3))
        for i, (sensor_type, pos) in enumerate(XELA_IMG_ORDER.items()):
            if "aftc" in sensor_type:
                tactile_image[pos[0] : pos[0] + 6, pos[1] : pos[1] + 6] = tactile_values[i]
            elif "4x4" in sensor_type:
                tactile_image[pos[0] : pos[0] + 4, pos[1]+2 : pos[1]+2 + 4] = tactile_values[i]
            elif "4x6" in sensor_type:
                tactile_image[pos[0] : pos[0] + 4, pos[1] : pos[1] + 6] = tactile_values[i]

        tactile_image = torch.FloatTensor(tactile_image)
        tactile_image = torch.permute(tactile_image, (2, 0, 1))
        return self.transform(tactile_image)


    def get_whole_hand_tactile_image_tdex(self, tactile_values):
        # tactile_values: (15,16,3) - turn it into 16,16,3 by concatenating 0z
        tactile_image = torch.FloatTensor(tactile_values)
        tactile_image = F.pad(tactile_image, (0, 0, 0, 0, 1, 0), "constant", 0)
        # reshape it to 4x4
        tactile_image = tactile_image.view(16, 4, 4, 3)

        pad_idx = list(range(16))
        if self.shuffle_type == "pad":
            random.seed(10)
            random.shuffle(pad_idx)

        tactile_image = torch.concat(
            [torch.concat([tactile_image[pad_idx[i * 4 + j]] for j in range(4)], dim=0) for i in range(4)],
            dim=1,
        )

        if self.shuffle_type == "whole":
            copy_tactile_image = copy(tactile_image)
            sensor_idx = list(range(16 * 16))
            random.seed(10)
            random.shuffle(sensor_idx)
            for i in range(16):
                for j in range(16):
                    rand_id = sensor_idx[i * 16 + j]
                    rand_i = int(rand_id / 16)
                    rand_j = int(rand_id % 16)
                    tactile_image[i, j, :] = copy_tactile_image[rand_i, rand_j, :]

        tactile_image = torch.permute(tactile_image, (2, 0, 1))

        return self.transform(tactile_image)

    def get_tactile_image_for_visualization(self, tactile_values):
        tactile_image = self._get_whole_hand_tactile_image(tactile_values)
        tactile_image = T.Resize(self.size, antialias=True)(tactile_image)  # Don't need another normalization
        tactile_image = (tactile_image - tactile_image.min()) / (tactile_image.max() - tactile_image.min())
        return tactile_image


def dump_tactile_state(tactile_values):
    tactile_coordinates = []
    for j in range(48, 192 + 1, 48):  # Y
        for i in range(48, 192 + 1, 48):  # X - It goes from top left to bottom right row first
            tactile_coordinates.append([i, j])
    tactile_coordinates = np.array(tactile_coordinates)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    for col_id in range(4):
        for row_id in range(4):
            if col_id + row_id > 0:
                _ = get_xela_heatmap(
                    axs[row_id, col_id],
                    magfield=tactile_values[col_id * 4 + row_id - 1],
                    coords=tactile_coordinates,
                    resolution=128,
                    height=240,
                    width=240,
                    scale=1,
                )

            axs[row_id, col_id].get_yaxis().set_ticks([])
            axs[row_id, col_id].get_xaxis().set_ticks([])
    return fig


def get_xela_heatmap(ax, magfield, coords, resolution, height, width, scale=1):
    strength = np.linalg.norm(magfield, axis=-1)
    direction = magfield / (strength[..., None] + 1e-6)
    direction = direction * 128

    scale_x = resolution / width
    scale_y = resolution / height

    assert len(strength) == len(coords)
    x, y = np.linspace(0, width, resolution), np.linspace(0, height, resolution)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.dstack([xx, yy]).reshape(-1, 2)
    zz = []
    for i, coord in enumerate(coords):
        cov = np.array([[width, 0], [0, height]]) * strength[i] * 0.25
        var = multivariate_normal(mean=coord, cov=cov)
        zz.append(var.pdf(xxyy))
    zz = np.sum(np.array(zz), axis=0)
    image = zz.reshape((resolution, resolution))
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image_plasma = matplotlib.colormaps.get_cmap("magma")(image) * 255
    image_plasma = image_plasma.astype(np.uint8)

    ax.imshow(image_plasma, extent=[0, resolution, 0, resolution])

    ax.scatter(
        coords[:, 0] * scale_x,
        coords[:, 1] * scale_y,
        c="cyan",
        s=15,
    )
    ax.quiver(
        coords[:, 0] * scale_x,
        coords[:, 1] * scale_y,
        direction[:, 0],
        direction[:, 1],
        angles="xy",
        scale_units="xy",
        scale=4,
        color="cyan",
    )

    return image_plasma
