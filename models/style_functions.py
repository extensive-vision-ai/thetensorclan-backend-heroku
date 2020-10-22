from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import Logger

import torch
from torch.jit import RecursiveScriptModule
from PIL import Image
import torchvision.transforms as T
from torch import Tensor
from numpy import uint8

from utils import setup_logger

logger: Logger = setup_logger(__name__)


def fast_style_transfer(
    model: RecursiveScriptModule, image: Image.Image
) -> Image.Image:
    H, W = image.size
    larger_side: int = H if H > W else W
    scale: float = larger_side / 500 if larger_side > 500 else 1.0

    content_image = image.resize((int(H / scale), int(W / scale)), Image.ANTIALIAS)

    transforms = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.mul(255))])

    img_tensor: Tensor = transforms(content_image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output: Tensor = model(img_tensor)
        output = output.squeeze(0)

    # convert [C, H, W] -> [H, W, C] -> numpy uint8 [0-255]
    img_np: uint8 = output.cpu().permute(1, 2, 0).clamp(0, 255).numpy().astype("uint8")

    return Image.fromarray(img_np)
