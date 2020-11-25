from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from functools import partial
from pathlib import Path
from typing import Union, Callable, Optional
from abc import ABC, abstractmethod

import pickle
import gdown
import requests

# noinspection PyUnresolvedReferences
import torch

# noinspection PyUnresolvedReferences
import torchtext
from requests.models import Response
from torch.jit import RecursiveScriptModule
from functools import wraps

from .autoencoder_functions import *
from .classifier_functions import *
from .generator_functions import *
from .text_functions import *
from .style_functions import *
from .speech_functions import *

from .translator_models import annotated_encoder_decoder_de_en

logger: Logger = setup_logger(__name__)


def get_temp_folder():
    environ = "heroku"

    if environ is "heroku":
        return Path("/tmp")

    # local
    return Path("tmp")


class Downloadable(ABC):
    def __init__(self, file_name: str):
        self.file_name: str = file_name

    @abstractmethod
    def download(self):
        raise NotImplementedError


class GithubRawFile(Downloadable):
    def __init__(self, file_name: str, file_url: str):
        super(GithubRawFile, self).__init__(file_name)
        self.file_url: str = file_url

    def download(self):
        logger.info(
            f"=> Downloading GithubRawFile {self.file_name} from {self.file_url}"
        )

        # heroku gives you `/tmp` to store files, which can be cached
        file_path: Path = get_temp_folder() / f"{self.file_name}"
        if not file_path.exists():
            r: Response = requests.get(self.file_url)
            with file_path.open("wb") as f:
                f.write(r.content)  # write the contents to file

        return file_path


class GoogleDriveFile(Downloadable):
    def __init__(self, file_name, file_id):
        super(GoogleDriveFile, self).__init__(file_name)
        self.file_id = file_id

    @property
    def gdrive_url(self):
        return f"https://drive.google.com/uc?id={self.file_id}"

    def download(self):
        logger.info(
            f"=> Downloading GDrive file {self.file_name} from {self.gdrive_url}"
        )

        file_path: Path = get_temp_folder() / f"{self.file_name}"
        if not file_path.exists():
            gdown.cached_download(url=self.gdrive_url, path=file_path)

        return file_path


"""
The model file must have .pt extension
The class file must have .json extension
But the extensions are not explicitly mentioned here in the REGISTER

Also we have different types of models, the 'type' specifies the type of model
"""
MODEL_REGISTER: Dict[str, Dict[str, Union[str, Any]]] = {
    "resnet34-imagenet": {
        "type": "classification",
        "model_file": "mobilenetv2_imagenet",
        "class_file": "imagenet_classes",
        "model_url": "https://drive.google.com/uc?id=1fqEXHD5fsqccSRjuIBTs8pJ8CLpp5FVz",
        "classifier_func": classify_resnet34_imagenet,
    },
    "mobilenetv2-ifo": {
        "type": "classification",
        "model_file": "mobilenetv2_ifo.traced",
        "class_file": "ifo_classes",
        "model_url": "https://drive.google.com/uc?id=1x130XEWyHBRy6Xbc4QHCH_Me1NXSicjj",
        "classifier_func": classify_mobilenetv2_ifo,
    },
    "indian-face": {
        "type": "classification",
        "model_file": "indian_face_model.traced",
        "class_file": "indian_face_classes",
        "model_url": "https://drive.google.com/uc?id=1jYVTVzY7PQZq2L-pbYApk42aXlv5XXKP",
        "classifier_func": classify_indian_face,
    },
    "lfw-plus": {
        "type": "classification",
        "model_file": "lfw_plus_model.traced",
        "class_file": "lfw_plus_classnames",
        "model_url": "https://drive.google.com/uc?id=1nFBXvf4eRiSmp_Wv_fUE_fDd7A9BJUDl",
        "classifier_func": classify_lfw_plus,
    },
    "red-car-gan-generator": {
        "type": "gan-generator",
        "latent_z_size": 64,
        "model_file": "red_car_gan_generator.traced",
        "model_url": "https://drive.google.com/uc?id=1mAJii2AljsY00c-4VkETDyyjgh_K0nNv",
        "generator_func": generate_red_car_gan,
    },
    "red-car-autoencoder": {
        "type": "variational-autoencoder",
        "latent_z_size": 2048,
        "model_file": "redcar_vae_128x128.traced",
        "model_url": "https://drive.google.com/uc?id=14lp_ZcLu--vwlITB3zYZCjQKJIwuQ55P",
        "autoencoder_func": autoencode_red_car,
    },
    "ifo-sr-gan": {
        "type": "gan-generator",
        "input_shape": (3, 200, 200),
        "model_file": "ifo_sr_model.traced",
        "model_url": "https://drive.google.com/uc?id=1MY5jUBN-dWssYbIU6sgQDBu2I70i-nik",
        "generator_func": generate_ifo_sr_gan,
    },
    "conv-sentimental-mclass": {
        "type": "text-sentiment",
        "vocab_file": "conv-sentimental-vocab",
        "vocab_url": "https://drive.google.com/uc?id=17Se87VyWyFpmvCy56R92k6McUL6pb_u7",
        "model_file": "conv-sentimental-mclass.scripted",
        "model_url": "https://drive.google.com/uc?id=1vJ7OLV5Y-xTX_xL3C0QgdT6qhmjIF0cj",
        "text_func": classify_conv_sentimental_mclass,
    },
    "fast-style-transfer": {
        "type": "style-transfer",
        "model_stack": {
            "candy": GoogleDriveFile(
                file_name="candy.scripted.pt",
                file_id="14W66DZbu8dnoWME-9Bs82yNBnqg059MJ",
            ),
            "mosaic": GoogleDriveFile(
                file_name="mosaic.scripted.pt",
                file_id="1_wgJZNHpJmI9n3A1nvps97jlPPUgF6R2",
            ),
            "rain_princess": GoogleDriveFile(
                file_name="rain_princess.scripted.pt",
                file_id="1Xr83ZZc3qtTJhL3mUNUcJkHFZ8boJqJg",
            ),
            "udnie": GoogleDriveFile(
                file_name="udnie.scripted.pt",
                file_id="1BFvOzva_ka2sIczs_5Hm5ByuUX54nWdm",
            ),
        },
        "transfer_func": fast_style_transfer,
    },
    "annotated-encoder-decoder-de-en": {
        "type": "translate-de-en",
        "metadata": GoogleDriveFile(
            file_name="annotated-encoder-decoder-de-en-meta.dill.pkl",
            file_id="1G4HuvGrgUMAUB8Qp_bqoLaKQNzA_yZbt",
        ),
        "model": GoogleDriveFile(
            file_name="annotated-encoder-decoder-de-en.pt",
            file_id="10oPjInWl0kH8kITq6HtZUgdxjf1TS0-e",
        ),
        "translate_func": translate_annotated_encoder_decoder_de_en,
    },
    "flickr8k-image-caption": {
        "type": "image-caption",
        "encoder": GoogleDriveFile(
            file_name="flickr8k_caption.encoder.scripted.pt",
            file_id="1x8p0vjKBnKjCDyBLm8GhC2IHIGCn6Lr7",
        ),
        "decoder": GoogleDriveFile(
            file_id="1wX4dJhr-i0spqRj5ge7I_IhngpRbNTFu",
            file_name="flickr8k_caption.decoder.scripted.pt",
        ),
        "wordmap": GoogleDriveFile(
            file_name="WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json",
            file_id="1ZfIO5rKq06c3UFuokISAcoc7O_bYuDJW",
        ),
        "caption_func": flickr8k_image_captioning,
    },
    "speech-recognition-residual-model": {
        "type": "speech-to-text",
        "model": GoogleDriveFile(
            file_name="speech-recognition-residual-model.scripted.pt",
            file_id="12MmbJioSAA-hs5y-RpHF0-Al5nAHdcgG",
        ),
        "speech_function": speech_recognition_residual_text,
    },
}


def load_meta_pkl(path):
    import pickle

    inp = open(path, "rb")
    meta = pickle.load(inp)
    inp.close()

    return meta


def load_meta_dill(path):
    import dill

    inp = open(path, "rb")
    meta = dill.load(inp)
    inp.close()

    return meta


def download_and_load_vocab(model_files) -> Dict[str, Vocab]:
    if "PRODUCTION" in os.environ:
        logger.info(
            f"=> Downloading Vocab file {model_files['vocab_file']} from {model_files['vocab_url']}"
        )

        # heroku gives you `/tmp` to store files, which can be cached
        vocab_path: Path = Path("/tmp") / f"{model_files['vocab_file']}.pkl"
        if not vocab_path.exists():
            gdown.cached_download(url=model_files["vocab_url"], path=vocab_path)

        logger.info(f"=> Loading {model_files['vocab_file']} from download_cache")
        with vocab_path.open("rb") as vocab_file:
            vocab: Dict[str, Vocab] = pickle.load(vocab_file)
    else:
        logger.info(f"=> Loading {model_files['vocab_file']} from Local")
        vocab_path: Path = Path("models") / (model_files["vocab_file"] + ".pkl")
        with vocab_path.open("rb") as vocab_file:
            vocab: Dict[str, Vocab] = pickle.load(vocab_file)

    return vocab


def download_and_load_model(model_files) -> RecursiveScriptModule:
    """

    Downloads and torch.jit.load the model from google drive, the downloaded model is saved in /tmp
        since in heroku we get /tmp to save all our stuff, if the app is not running in production
        the model must be saved in load storage, hence the model is directly loaded

    Args:
        model_files: the dict containing the model information

    Returns:
        (RecursiveScriptModule): the loaded torch.jit model
    """
    if "PRODUCTION" in os.environ:
        logger.info(
            f"=> Downloading Model {model_files['model_file']} from {model_files['model_url']}"
        )

        # heroku gives you `/tmp` to store files, which can be cached
        model_path: Path = Path("/tmp") / f"{model_files['model_file']}.pt"
        if not model_path.exists():
            gdown.cached_download(url=model_files["model_url"], path=model_path)

        logger.info(f"=> Loading {model_files['model_file']} from download_cache")
        model: RecursiveScriptModule = torch.jit.load(str(model_path))
    else:
        logger.info(f"=> Loading {model_files['model_file']} from Local")
        model = torch.jit.load(
            str((Path("models") / (model_files["model_file"] + ".pt")))
        )

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

    classes_list: List[str] = json.load(
        open(Path("models") / (model_files["class_file"] + ".json"))
    )

    model: RecursiveScriptModule = download_and_load_model(model_files)

    classifier: Callable[
        [RecursiveScriptModule, List[str], Image.Image], List[Dict[str, Any]]
    ] = model_files["classifier_func"]

    return partial(classifier, model, classes_list)


def get_generator(model_name) -> Callable[[Union[np.ndarray, Image.Image]], Any]:
    """
    get_generator

    fetches the generator function for the given model from the MODEL_REGISTER

    Args:
        model_name: the model name registered in MODEL_REGISTER

    Returns:
       (Callable[[np.ndarray], Any]): a callable function that takes the input latent_z
            and returns the generated image
    """
    model_files: Dict[str, Union[str, Any]] = MODEL_REGISTER[model_name]

    model: RecursiveScriptModule = download_and_load_model(model_files)

    generator_func: Callable[
        [RecursiveScriptModule, Union[np.ndarray, Image.Image]], Any
    ] = model_files["generator_func"]

    return partial(generator_func, model)


def get_autoencoder(
    model_name: str,
) -> Callable[[Image.Image], Tuple[Image.Image, np.ndarray]]:
    """
    get_autoencoder

    fetches the generator function for the given model from the MODEL_REGISTER, downloads the model
        from google drives, and returns a partially applied function that only needs the input to the model
        and the function gives the output of the model, in a required way by this backend

    Args:
        model_name: the model name registered in MODEL_REGISTER

    Returns:
       (Callable[[np.ndarray], Tuple[Image.Image, np.ndarray]]): a callable function that takes the input latent_z
            and returns the generated image and the encoded latent_z numpy array
    """
    model_files: Dict[str, Union[str, Any]] = MODEL_REGISTER[model_name]

    model: RecursiveScriptModule = download_and_load_model(model_files)

    generator_func: Callable[[RecursiveScriptModule, np.ndarray], Any] = model_files[
        "autoencoder_func"
    ]

    return partial(generator_func, model)


def get_text_function(model_name: str):
    model_files = MODEL_REGISTER[model_name]

    model: RecursiveScriptModule = download_and_load_model(model_files)
    vocab: Dict[str, Vocab] = download_and_load_vocab(model_files)

    text_func = model_files["text_func"]

    return partial(text_func, model, vocab)


def get_text_translate_function(model_name: str) -> Callable[[str], str]:
    model_files = MODEL_REGISTER[model_name]

    if model_name == "annotated-encoder-decoder-de-en":
        meta_file: GoogleDriveFile = model_files["metadata"]
        model_file: GoogleDriveFile = model_files["model"]

        meta: Dict[str, Any] = load_meta_dill(meta_file.download())
        model_state: Dict[str, Tensor] = torch.load(
            model_file.download(), map_location="cpu"
        )

        logger.info("=> Loading annotated_encoder_decoder_de_en.EncoderDecoder")

        model: annotated_encoder_decoder_de_en.EncoderDecoder = (
            annotated_encoder_decoder_de_en.make_model(
                len(meta["SRC.vocab.itos"]),
                len(meta["TRG.vocab.itos"]),
                emb_size=256,
                hidden_size=256,
                num_layers=1,
                dropout=0.2,
            )
        )
        model.load_state_dict(model_state)

        translate_func: Callable[
            [annotated_encoder_decoder_de_en.EncoderDecoder, Dict[str, Any], str], str
        ] = model_files["translate_func"]

        @wraps(translate_func)
        def wrapper(source_text: str) -> str:
            return translate_func(model, meta, source_text)

        return wrapper

    else:
        raise Exception(f"UNKNOWN {model_name}")


def get_style_transfer_function(
    model_name: str, style_name: str
) -> Callable[[Image.Image], Image.Image]:
    model_files: Dict[str, Any] = MODEL_REGISTER[model_name]["model_stack"]

    model_file: GoogleDriveFile = model_files[style_name]

    model: RecursiveScriptModule = torch.jit.load(str(model_file.download()))

    transfer_func: Callable[
        [RecursiveScriptModule, Image.Image], Image.Image
    ] = MODEL_REGISTER[model_name]["transfer_func"]

    # a drop in replacement for partial
    # fixes the type hint inference problem in partial
    @wraps(transfer_func)
    def wrapper(image: Image.Image) -> Image.Image:
        return transfer_func(model, image)

    return wrapper


def get_image_captioning_function(model_name: str) -> Callable[[Image.Image], str]:
    model_files: Dict[str, Any] = MODEL_REGISTER[model_name]

    encoder_file: GoogleDriveFile = model_files["encoder"]
    decoder_file: GoogleDriveFile = model_files["decoder"]
    wordmap_file: GoogleDriveFile = model_files["wordmap"]

    encoder: RecursiveScriptModule = torch.jit.load(
        str(encoder_file.download()), map_location="cpu"
    )

    decoder: RecursiveScriptModule = torch.jit.load(
        str(decoder_file.download()), map_location="cpu"
    )

    with wordmap_file.download().open("r") as f:
        word_map: Dict[str, int] = json.load(f)

    caption_func: Callable[
        [
            RecursiveScriptModule,
            RecursiveScriptModule,
            Dict[str, int],
            Image.Image,
            Optional[int],
        ],
        str,
    ] = model_files["caption_func"]

    @wraps(caption_func)
    def wrapper(image: Image.Image) -> str:
        with torch.no_grad():
            return caption_func(encoder, decoder, word_map, image, 3)

    return wrapper


def get_speech_to_text_function(model_name: str) -> Callable[[Tensor], str]:
    model_files: Dict[str, Any] = MODEL_REGISTER[model_name]

    model_file: GoogleDriveFile = model_files["model"]

    model: RecursiveScriptModule = torch.jit.load(
        str(model_file.download()), map_location="cpu"
    )

    speech_to_text_function: Callable[
        [RecursiveScriptModule, Tensor], str
    ] = model_files["speech_function"]

    @wraps(speech_to_text_function)
    def wrapper(input_audio: Tensor) -> str:
        return speech_to_text_function(model, input_audio)

    return wrapper
