from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import Logger

import numpy as np
import torch
from PIL import Image
from numpy import uint8
from torch import Tensor
from torch.jit import RecursiveScriptModule
import torchvision.transforms as T

from utils import setup_logger

logger: Logger = setup_logger(__name__)


def norm_ip_(img, min, max) -> None:
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)


def norm_tensor_(t) -> None:
    """
    Performs Min-Max Normalization "In-Place" for the input tensor

    Args:
        t: input tensor
    """
    norm_ip_(t, float(t.min()), float(t.max()))


def generate_red_car_gan(
    model: RecursiveScriptModule, latent_z: np.ndarray
) -> Image.Image:
    """
    generate_red_car_gan

        a generator function for the Red Car GAN, which takes in a latent z vector
        and generates a red car based on those values

    Args:
        model: the red car gan traced and loaded generator model
        latent_z: the latent z vector for the generator

    Returns:
        Image: The generated image from the model

    """

    latent_z: Tensor = torch.tensor(latent_z)
    latent_z = latent_z.reshape(-1, 1, 1)
    latent_z = latent_z.unsqueeze(0)

    with torch.no_grad():
        fake: Tensor = model(latent_z)
        fake = fake.squeeze(0)

    # normalize the values to [0, 1]
    norm_tensor_(fake)
    # to change from C, H, W -> H, W, C
    fake: Tensor = fake.permute(1, 2, 0)

    # convert it to uint8 array for PIL to create a Image out of it
    fake_img_arr: uint8 = np.uint8((fake.numpy() * 255).astype(int))

    return Image.fromarray(fake_img_arr)


def generate_ifo_sr_gan(
    model: RecursiveScriptModule, image: Image.Image
) -> Image.Image:

    trans: T.Compose = T.Compose(
        [
            T.Resize((200, 200)),
            T.ToTensor(),
        ]
    )

    img_tensor: Tensor = trans(image).unsqueeze(0)

    with torch.no_grad():
        sr_image: Tensor = model(img_tensor)
        sr_image = sr_image.squeeze(0)

    # to change from C, H, W -> H, W, C
    sr_image: Tensor = sr_image.permute(1, 2, 0)

    # convert it to uint8 array for PIL to create a Image out of it
    fake_img_arr: uint8 = np.uint8((sr_image.numpy() * 255).astype(int))

    return Image.fromarray(fake_img_arr)
