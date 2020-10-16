from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.jit import RecursiveScriptModule
from torchvision.transforms import Compose

from models.utils import UnNormalize


def autoencode_red_car(
    model: RecursiveScriptModule, image: Image.Image
) -> Tuple[Image.Image, np.ndarray]:
    """
    autoencode_red_car

        An autoencoder function for RedCarVAE, this takes in the m

    Args:
        model: the model that will be used for inferencing
        image: the image which is to be encoded and reconstructed

    Returns:
        (Tuple[Image.Image, Tensor]): the reconstructed image and the latent_z vector representation of the image
    """
    trans: Compose = T.Compose(
        [
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.570838093757629, 0.479552984237671, 0.491760671138763],
                std=[0.279659748077393, 0.309973508119583, 0.311098515987396],
            ),
        ]
    )

    unorm: UnNormalize = UnNormalize(
        mean=[0.570838093757629, 0.479552984237671, 0.491760671138763],
        std=[0.279659748077393, 0.309973508119583, 0.311098515987396],
    )

    img_tensor: Tensor = trans(image).unsqueeze(0)

    with torch.no_grad():
        reconstructed_x: Tensor
        mu: Tensor

        reconstructed_x, mu, _ = model(img_tensor)
        reconstructed_x.squeeze_(0)
        mu.squeeze_(0)

    # un-normalize the values, the model has learnt normalize values and predicts
    # thus
    reconstructed_x: Tensor = unorm(reconstructed_x)

    # convert C, H, W -> H, W, C
    reconstructed_x: Tensor = reconstructed_x.permute(1, 2, 0)

    reconstructed_x_arr: np.uint8 = np.uint8(
        (reconstructed_x.numpy() * 255).astype(int)
    )

    # return the reconstructed image and the mean of encoded values (probability distr.)
    return Image.fromarray(reconstructed_x_arr), mu.reshape((-1,)).cpu().numpy()
