from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import gdown

from utils import setup_logger

logger = setup_logger(__name__)

from operator import itemgetter

import numpy as np
import copy
import os.path as osp
import re
import os
import sys
import cv2
from PIL import Image
import torchvision.transforms as T
import torch


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# add the poselib to path
this_dir = osp.dirname(__file__)
pose_lib_path = osp.join(this_dir, '..', 'helper_repositories/human_pose_estimation_pytorch')
add_path(pose_lib_path)

RESNET_50_256x256 = 'https://drive.google.com/uc?id=1V2AaVpDSn-eS7jrFScHLJ-wvTFuQ0-Dc'
CONFIG_FILE = osp.join(this_dir, '..',
                       'helper_repositories/human_pose_estimation_pytorch/experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml')
MODEL_PATH = '/tmp/pose_resnet_50_256x256.pth.tar'


# download and setup the model
def setup_model():
    if 'PRODUCTION' in os.environ:
        # heroku gives the /tmp directory for temporary files
        MODEL_PATH = '/tmp/pose_resnet_50_256x256.pth.tar'
    else:
        MODEL_PATH = './pose_resnet_50_256x256.pth.tar'
    # download the model
    if not os.path.exists(MODEL_PATH):
        logger.info(f'Downloading {MODEL_PATH}')
        gdown.download(url=RESNET_50_256x256, output=MODEL_PATH)
    # gdown.cached_download(url=RESNET_50_256x256, path=MODEL_PATH)


setup_model()

# import poselib libraries
from pose_lib.core.config import config
from pose_lib.core.config import update_config
import pose_lib.models as pmodels

import torch

update_config(CONFIG_FILE)
config.GPUS = ''

logger.info('=> Loading the Pose Model')
model = eval('pmodels.' + config.MODEL.NAME + '.get_pose_net')(config, is_train=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))


JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']
JOINTS = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in JOINTS]

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
              [14, 15]
]


def get_detached(x: torch.Tensor):
    return copy.deepcopy(x.cpu().detach().numpy())


get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])


def get_pose(image: Image.Image) -> Image.Image:
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tr_img: torch.Tensor = transform(image)

    output: torch.Tensor = model(tr_img.unsqueeze(0))
    output = output.squeeze(0)
    _, OUT_HEIGHT, OUT_WIDTH = output.shape

    pose_layers = get_detached(x=output)
    key_points = list(get_keypoints(pose_layers=pose_layers))

    return apply_pose_to_image(image=image, key_points=key_points, out_shape=(OUT_HEIGHT, OUT_WIDTH))


def apply_pose_to_image(image: Image.Image, key_points, out_shape: Tuple[int, int] = (64, 64), thr: float = 0.8):
    image_p = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    is_joint_plotted = [False for _ in range(len(JOINTS))]

    for pose_pair in POSE_PAIRS:
        from_j, to_j = pose_pair

        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]

        IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape

        from_x_j, to_x_j = from_x_j * IMG_WIDTH / out_shape[0], to_x_j * IMG_WIDTH / out_shape[0]
        from_y_j, to_y_j = from_y_j * IMG_HEIGHT / out_shape[1], to_y_j * IMG_HEIGHT / out_shape[1]

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        if from_thr > thr and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(image_p, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True

        if to_thr > thr and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(image_p, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        if from_thr > thr and to_thr > thr:
            # this is a joint connection, plot a line
            cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)

    return Image.fromarray(cv2.cvtColor(image_p, cv2.COLOR_BGR2RGB))
