from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import Logger

import torch
from numpy import uint8
from torch.jit import RecursiveScriptModule
from  typing import List
from torch import Tensor
import numpy as np
from PIL import  Image

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


def generate_red_car_gan(model: RecursiveScriptModule, latent_z: np.ndarray) -> Image.Image:
    latent_z: Tensor = torch.tensor(latent_z)
    latent_z = latent_z.reshape(-1, 1, 1)
    latent_z = latent_z.unsqueeze(0)

    with torch.no_grad():
        fake: Tensor = model(latent_z)
        fake = fake.squeeze(0)

    # normalize the values to [0, 1]
    norm_tensor_(fake)
    # to change from C, H, W -> H, W, C
    fake = fake.permute(1, 2, 0)

    # convert it to uint8 array for PIL to create a Image out of it
    fake_img_arr: uint8 = np.uint8((fake.numpy() * 255).astype(int))

    return Image.fromarray(fake_img_arr)
