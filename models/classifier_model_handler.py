from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from functools import partial
from pathlib import Path
from typing import Union, Callable

import gdown
import torch

from .classifiers import *

logger: Logger = setup_logger(__name__)

'''
The model file must have .pt extension
The class file must have .json extension
But the extensions are not explicitly mentioned here in the REGISTER
'''
MODEL_REGISTER: Dict[str, Dict[str, Union[str, Any]]] = {
    'resnet34-imagenet': {
        'model_file': 'mobilenetv2_imagenet',
        'class_file': 'imagenet_classes',
        'model_url': 'https://drive.google.com/uc?id=1fqEXHD5fsqccSRjuIBTs8pJ8CLpp5FVz',
        'classifier_func': classify_resnet34_imagenet
    },
    'mobilenetv2-ifo': {
        'model_file': 'mobilenetv2_ifo.traced',
        'class_file': 'ifo_classes',
        'model_url': 'https://drive.google.com/uc?id=1x130XEWyHBRy6Xbc4QHCH_Me1NXSicjj',
        'classifier_func': classify_mobilenetv2_ifo
    },
    'indian-face': {
        'model_file': 'indian_face_model.traced',
        'class_file': 'indian_face_classes',
        'model_url': 'https://drive.google.com/uc?id=1jYVTVzY7PQZq2L-pbYApk42aXlv5XXKP',
        'classifier_func': classify_indian_face
    },
    'lfw-plus': {
        'model_file': 'lfw_plus_model.traced',
        'class_file': 'lfw_plus_classnames',
        'model_url': 'https://drive.google.com/uc?id=1nFBXvf4eRiSmp_Wv_fUE_fDd7A9BJUDl',
        'classifier_func': classify_lfw_plus
    }
}


def get_classifier(model_name) -> partial[List[Dict[str, Any]]]:
    """

    Args:
        model_name: the model_handel name, must be in MODEL_REGISTER

    Returns:
        (partial[List[Dict[str, Any]]]: a callable function that just needs the input image and it returns results
    """
    model_files = MODEL_REGISTER[model_name]

    classes_list: List[str] = json.load(open(Path('models') / (model_files['class_file'] + '.json')))

    if 'PRODUCTION' in os.environ:
        logger.info(f"=> Downloading Model {model_files['model_file']} from {model_files['model_url']}")

        # heroku gives you `/tmp` to store files, which can be cached
        model_path: Path = Path('/tmp') / f"{model_files['model_file']}.pt"
        gdown.cached_download(url=model_files['model_url'], path=model_path)

        logger.info(f"=> Loading {model_files['model_file']} from download_cache")
        model: RecursiveScriptModule = torch.jit.load(str(model_path))
    else:
        logger.info(f"=> Loading {model_files['model_file']} from Local")
        model = torch.jit.load(str((Path('models') / (model_files['model_file'] + '.pt'))))

    classifier: Callable[[RecursiveScriptModule, List[str], Image.Image], List[Dict[str, Any]]] = \
        model_files['classifier_func']

    return partial(classifier, model, classes_list)
