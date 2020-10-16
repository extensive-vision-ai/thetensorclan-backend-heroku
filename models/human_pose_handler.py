from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple, Dict

import gdown
from onnxruntime import InferenceSession

from utils import setup_logger

logger = setup_logger(__name__)

from operator import itemgetter

import numpy as np
import copy
import os.path as osp
import re
from pathlib import Path
import os
import sys
import cv2
from PIL import Image
import torchvision.transforms as T
import torch
import onnxruntime

# model gdrive download url
RESNET_50_256x256_ONNX_QUANT: Dict[str, str] = {
    "model_file": "pose_resnet_50_256x256.quantized.onnx",
    "model_url": "https://drive.google.com/uc?id=1CAefkMUP7ww_cq9MzBrMzA_INzSWaWdU",
}


# download and setup the model
def get_ort_model() -> InferenceSession:
    if "PRODUCTION" in os.environ:
        # heroku gives the /tmp directory for temporary files
        model_path: Path = (
            Path("/tmp") / f"{RESNET_50_256x256_ONNX_QUANT['model_file']}"
        )
    else:
        model_path: Path = Path("./") / f"{RESNET_50_256x256_ONNX_QUANT['model_file']}"

    if not model_path.exists():
        logger.info(f"Downloading Model : {RESNET_50_256x256_ONNX_QUANT['model_file']}")
        gdown.cached_download(
            url=RESNET_50_256x256_ONNX_QUANT["model_url"], path=model_path
        )

    ort_session: InferenceSession = onnxruntime.InferenceSession(str(model_path))

    return ort_session


# some MPII specific stuff
JOINTS = [
    "0 - r ankle",
    "1 - r knee",
    "2 - r hip",
    "3 - l hip",
    "4 - l knee",
    "5 - l ankle",
    "6 - pelvis",
    "7 - thorax",
    "8 - upper neck",
    "9 - head top",
    "10 - r wrist",
    "11 - r elbow",
    "12 - r shoulder",
    "13 - l shoulder",
    "14 - l elbow",
    "15 - l wrist",
]
JOINTS = [re.sub(r"[0-9]+|-", "", joint).strip().replace(" ", "-") for joint in JOINTS]

POSE_PAIRS = [
    # UPPER BODY
    [9, 8],
    [8, 7],
    [7, 6],
    # LOWER BODY
    [6, 2],
    [2, 1],
    [1, 0],
    [6, 3],
    [3, 4],
    [4, 5],
    # ARMS
    [7, 12],
    [12, 11],
    [11, 10],
    [7, 13],
    [13, 14],
    [14, 15],
]


def get_detached(x: torch.Tensor):
    return copy.deepcopy(x.cpu().detach().numpy())


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


get_keypoints = lambda pose_layers: map(
    itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers]
)


def get_pose(image: Image.Image) -> Image.Image:
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tr_img: torch.Tensor = transform(image)

    ort_model: InferenceSession = get_ort_model()

    ort_inputs = {ort_model.get_inputs()[0].name: to_numpy(tr_img.unsqueeze(0))}
    ort_outs = ort_model.run(None, ort_inputs)
    output = np.array(ort_outs[0][0])

    _, OUT_HEIGHT, OUT_WIDTH = output.shape

    pose_layers = output
    key_points = list(get_keypoints(pose_layers=pose_layers))

    return apply_pose_to_image(
        image=image, key_points=key_points, out_shape=(OUT_HEIGHT, OUT_WIDTH)
    )


def apply_pose_to_image(
    image: Image.Image,
    key_points,
    out_shape: Tuple[int, int] = (64, 64),
    thr: float = 0.5,
):
    image_p = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    is_joint_plotted = [False for _ in range(len(JOINTS))]

    for pose_pair in POSE_PAIRS:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]

        IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape

        from_x_j, to_x_j = (
            from_x_j * IMG_WIDTH / out_shape[0],
            to_x_j * IMG_WIDTH / out_shape[0],
        )
        from_y_j, to_y_j = (
            from_y_j * IMG_HEIGHT / out_shape[1],
            to_y_j * IMG_HEIGHT / out_shape[1],
        )

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        if from_thr > thr and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(
                image_p,
                (from_x_j, from_y_j),
                (4, 4),
                0,
                0,
                360,
                (255, 255, 255),
                cv2.FILLED,
            )
            is_joint_plotted[from_j] = True

        if to_thr > thr and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(
                image_p,
                (to_x_j, to_y_j),
                (4, 4),
                0,
                0,
                360,
                (255, 255, 255),
                cv2.FILLED,
            )
            is_joint_plotted[to_j] = True

        if from_thr > thr and to_thr > thr:
            # this is a joint connection, plot a line
            cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)

    return Image.fromarray(cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB))
