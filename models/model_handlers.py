from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from functools import partial
from pathlib import Path
from typing import Union, Callable, Dict, Any, List

import gdown
import torch
from torch.jit import RecursiveScriptModule

from .classifier_functions import *
from .generator_functions import *

logger: Logger = setup_logger(__name__)

'''
The model file must have .pt extension
The class file must have .json extension
But the extensions are not explicitly mentioned here in the REGISTER

Also we have different types of models, the 'type' specifies the type of model
'''
MODEL_REGISTER: Dict[str, Dict[str, Union[str, Any]]] = {
    'resnet34-imagenet': {
        'type': 'classification',
        'model_file': 'mobilenetv2_imagenet',
        'class_file': 'imagenet_classes',
        'model_url': 'https://drive.google.com/uc?id=1fqEXHD5fsqccSRjuIBTs8pJ8CLpp5FVz',
        'classifier_func': classify_resnet34_imagenet
    },
    'mobilenetv2-ifo': {
        'type': 'classification',
        'model_file': 'mobilenetv2_ifo.traced',
        'class_file': 'ifo_classes',
        'model_url': 'https://drive.google.com/uc?id=1x130XEWyHBRy6Xbc4QHCH_Me1NXSicjj',
        'classifier_func': classify_mobilenetv2_ifo
    },
    'indian-face': {
        'type': 'classification',
        'model_file': 'indian_face_model.traced',
        'class_file': 'indian_face_classes',
        'model_url': 'https://drive.google.com/uc?id=1jYVTVzY7PQZq2L-pbYApk42aXlv5XXKP',
        'classifier_func': classify_indian_face
    },
    'lfw-plus': {
        'type': 'classification',
        'model_file': 'lfw_plus_model.traced',
        'class_file': 'lfw_plus_classnames',
        'model_url': 'https://drive.google.com/uc?id=1nFBXvf4eRiSmp_Wv_fUE_fDd7A9BJUDl',
        'classifier_func': classify_lfw_plus
    },
    'red-car-gan-generator': {
        'type': 'gan-generator',
        'latent_z_size': 64,
        'model_file': 'red_car_gan_generator.traced',
        'model_url': 'https://drive.google.com/uc?id=1mAJii2AljsY00c-4VkETDyyjgh_K0nNv',
        'generator_func': generate_red_car_gan
    }
}


def download_and_loadmodel(model_files) -> RecursiveScriptModule:
    """

    Downloads and torch.jit.load the model from google drive, the downloaded model is saved in /tmp
        since in heroku we get /tmp to save all our stuff, if the app is not running in production
        the model must be saved in load storage, hence the model is directly loaded

    Args:
        model_files: the dict containing the model information

    Returns:
        (RecursiveScriptModule): the loaded torch.jit model
    """
    if 'PRODUCTION' in os.environ:
        logger.info(f"=> Downloading Model {model_files['model_file']} from {model_files['model_url']}")

        # heroku gives you `/tmp` to store files, which can be cached
        model_path: Path = Path('/tmp') / f"{model_files['model_file']}.pt"
        if not model_path.exists():
            gdown.cached_download(url=model_files['model_url'], path=model_path)

        logger.info(f"=> Loading {model_files['model_file']} from download_cache")
        model: RecursiveScriptModule = torch.jit.load(str(model_path))
    else:
        logger.info(f"=> Loading {model_files['model_file']} from Local")
        model = torch.jit.load(str((Path('models') / (model_files['model_file'] + '.pt'))))

    return model


def get_classifier(model_name) -> Callable[[Image.Image], List[Dict[str, Any]]]:
    """

    get_classifier

    returns a function handle for the model's classifier, the function needs only the input
        and it returns the classified class

    internally this is implemented by keeping `MODEL_REGISTER` that keeps track of the models available
        for inferencing, it also contains the corresponding function that is responsible to classify the
        input and also transforming the output of the model into classnames and the class confidences,
        this is used in the front end for displaying the results nicely

    Args:
        model_name: the model_handel name, must be in MODEL_REGISTER

    Returns:
        (partial[List[Dict[str, Any]]]: a callable function that just needs the input image and it returns results
    """
    model_files: Dict[str, Union[str, Any]] = MODEL_REGISTER[model_name]

    classes_list: List[str] = json.load(open(Path('models') / (model_files['class_file'] + '.json')))

    model: RecursiveScriptModule = download_and_loadmodel(model_files)

    classifier: Callable[[RecursiveScriptModule, List[str], Image.Image], List[Dict[str, Any]]] = \
        model_files['classifier_func']

    return partial(classifier, model, classes_list)


def get_generator(model_name) -> Callable[[np.ndarray], Any]:

    model_files: Dict[str, Union[str, Any]] = MODEL_REGISTER[model_name]

    model: RecursiveScriptModule = download_and_loadmodel(model_files)

    generator_func: Callable[[RecursiveScriptModule, np.ndarray], Any] = model_files['generator_func']

    return partial(generator_func, model)

