from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import Logger
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.jit import RecursiveScriptModule
from torchvision.transforms import Compose

from utils import setup_logger

logger: Logger = setup_logger(__name__)


def classify_resnet34_imagenet(
    model: RecursiveScriptModule, classes: List[str], image: Image.Image
) -> List[Dict[str, Any]]:
    trans: Compose = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor: Tensor = trans(image).unsqueeze(0)
    with torch.no_grad():
        predicted: Tensor = model(img_tensor).squeeze(0)
        predicted: Tensor = F.softmax(predicted)
    sorted_values = predicted.argsort(descending=True).cpu().numpy()

    top10pred: List[Dict[str, Any]] = list(
        map(
            lambda x: {
                "class_idx": x.item(),
                "class_name": classes[x],
                "confidence": predicted[x].item(),
            },
            sorted_values,
        )
    )[:10]

    return top10pred


def classify_mobilenetv2_ifo(
    model: RecursiveScriptModule, classes: List[str], image: Image.Image
) -> List[Dict[str, Any]]:
    trans: Compose = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.533459901809692, 0.584880530834198, 0.615305066108704],
                std=[0.172962218523026, 0.167985364794731, 0.184633478522301],
            ),
        ]
    )

    img_tensor = trans(image).unsqueeze(0)
    with torch.no_grad():
        predicted = model(img_tensor).squeeze(0)
        predicted = F.softmax(predicted)
    sorted_values = predicted.argsort(descending=True).cpu().numpy()

    top4pred: List[Dict[str, Any]] = list(
        map(
            lambda x: {
                "class_idx": x.item(),
                "class_name": classes[x],
                "confidence": predicted[x].item(),
            },
            sorted_values,
        )
    )[:4]

    return top4pred


def classify_indian_face(
    model: RecursiveScriptModule, classes: List[str], image: Image.Image
) -> List[Dict[str, Any]]:
    """
    The image sent to this MUST be aligned
    """
    trans: Compose = T.Compose(
        [
            T.Resize(160),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = trans(image).unsqueeze(0)
    with torch.no_grad():
        predicted = model(img_tensor).squeeze(0)
        predicted = F.softmax(predicted)
    sorted_values = predicted.argsort(descending=True).cpu().numpy()

    logger.info(sorted_values)

    top4pred: List[Dict[str, Any]] = list(
        map(
            lambda x: {
                "class_idx": x.item(),
                "class_name": classes[x],
                "confidence": predicted[x].item(),
            },
            sorted_values,
        )
    )[:4]

    return top4pred


def classify_lfw_plus(
    model: RecursiveScriptModule, classes: List[str], image: Image.Image
) -> List[Dict[str, Any]]:
    """
    The image sent to this MUST be aligned
    """
    trans: Compose = T.Compose(
        [
            T.Resize(160),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = trans(image).unsqueeze(0)
    with torch.no_grad():
        predicted = model(img_tensor).squeeze(0)
        predicted = F.softmax(predicted)
    sorted_values = predicted.argsort(descending=True).cpu().numpy()

    logger.info(sorted_values)

    top6pred: List[Dict[str, Any]] = list(
        map(
            lambda x: {
                "class_idx": x.item(),
                "class_name": classes[x],
                "confidence": predicted[x].item(),
            },
            sorted_values,
        )
    )[:6]

    return top6pred
