from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
from functools import wraps
from typing import Tuple, Any, Union, Callable

import numpy as np
from PIL import ImageFile
from PIL.Image import Image
from flask import Flask, jsonify, request, Response, make_response, Request
from flask_cors import CORS, cross_origin
from werkzeug.datastructures import FileStorage

from models import get_classifier
from models.model_handlers import (
    MODEL_REGISTER,
    get_generator,
    get_autoencoder,
    get_text_function,
    get_style_transfer_function,
    get_text_translate_function,
    get_image_captioning_function,
)
from utils import setup_logger, allowed_file, file2image
from utils.upload_utils import image2b64

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = setup_logger(__name__)
logger.info("=> Finished Importing")

# attach our logger to the system exceptions
sys.excepthook = lambda type, val, tb: logger.error(
    "Unhandled exception:", exc_info=val
)

app: Flask = Flask(__name__)
cors: CORS = CORS(app=app)
app.config["CORS_HEADERS"] = "Content-Type"

if "PRODUCTION" not in os.environ:
    app.config["DEBUG"] = True


@app.route("/")
@cross_origin()
def hello_thetensorclan() -> Tuple[Any, int]:
    return (
        jsonify({"message": "You've reached the TensorClan Heroku Backend EndPoint"}),
        200,
    )


@app.route("/classify/<model_handle>", methods=["POST"])
@cross_origin()
def classify_image_api(model_handle="resnet34-imagenet") -> Response:
    """

    Args:
        model_handle: the model handle string, should be in `models.model_handler.MODEL_REGISTER`

    Returns:
        (Response): if error then a json of {'error': 'message'} is sent
                    else return a json of sorted List[Dict[{'class_idx': idx, 'class_name': cn, 'confidence': 'c'}]]
    """
    if model_handle not in MODEL_REGISTER:
        return Response(
            {"error": f"{model_handle} not found in registered models"}, status=404
        )

    if "file" not in request.files:
        return Response({"error": "No file part"}, status=412)

    file: FileStorage = request.files["file"]

    if file.filename == "":
        return Response({"error": "No file selected"}, status=417)

    if allowed_file(file.filename):
        image: Image = file2image(file)
        classifier = get_classifier(model_handle)
        output = classifier(image)
        return Response(json.dumps(output), status=200)

    else:
        return Response({"error": f"{file.mimetype} not allowed"}, status=412)


@app.route("/generators/<model_handle>", methods=["POST"])
@cross_origin()
def generator_api(model_handle="red-car-gan-generator") -> Response:
    """
    generator_api

        This is the generator end point, that has the model handle as the parameter
            and takes in the latent_z values in the POST requests, followed by passing this
            vector to the model and generates an image, which is returned as a b64 image in
            the Response
    Args:
        model_handle: the model handle string

    Returns:
        Response: the base 64 encoded generated image

    """
    if model_handle not in MODEL_REGISTER:
        return make_response(
            jsonify({"error": f"{model_handle} not found in registered models"}), 404
        )

    if (
        model_handle in MODEL_REGISTER
        and MODEL_REGISTER[model_handle]["type"] != "gan-generator"
    ):
        return make_response(
            jsonify({"error": f"{model_handle} model is not a GAN"}), 412
        )

    if "latent_z_size" in MODEL_REGISTER[model_handle]:
        # this is a latentz input type of gan model
        if "latent_z" not in request.form:
            return make_response(
                jsonify({"error": "latent_z not found in the form"}), 412
            )

        latent_z = json.loads(f"[{request.form['latent_z']}]")
        latent_z = np.array(latent_z, dtype=np.float32)

        generator = get_generator(model_handle)
        output = generator(latent_z)

        # convert it to b64 bytes
        b64_image = image2b64(output)

        return make_response(jsonify(b64_image), 200)

    if "input_shape" in MODEL_REGISTER[model_handle]:
        # this is a image input type of gan model

        if "file" not in request.files:
            return make_response(jsonify({"error": "No file part"}), 412)

        file: FileStorage = request.files["file"]

        if file.filename == "":
            return make_response(jsonify({"error": "No file selected"}), 417)

        if allowed_file(file.filename):
            image: Image = file2image(file)
            generator = get_generator(model_handle)
            output = generator(image)

            # convert it to b64 bytes
            b64_image = image2b64(output)

            return make_response(jsonify(b64_image), 200)

    return make_response(jsonify({"error": f"{model_handle} is not a valid GAN"}), 412)


@app.route("/autoencoders/<model_handle>", methods=["POST"])
@cross_origin()
def autoencoder_api(model_handle="red-car-autoencoder") -> Response:
    """

    autoencoder_api

        This end point is used to encode an image and then get the latentz vector as well as
            the reconstructed image, this kind of technique can be used for image compression
            and video compression, but right now only supports images and specific type of input
            data.
        The latentz vector is a unique representation of the input, and thus the latentz given
            to a encoder and reconstruct the image exactly, thus reducing the data transmitted.

    Args:
        model_handle:  the model handle string, must be in the MODEL_REGISTER

    Returns:
        Response: The response is a JSON containing the reconstructed image and the latent z
            vector for the image

    """
    if model_handle not in MODEL_REGISTER:
        return make_response(
            jsonify({"error": f"{model_handle} not found in registered models"}), 404
        )

    if (
        model_handle in MODEL_REGISTER
        and MODEL_REGISTER[model_handle]["type"] != "variational-autoencoder"
    ):
        return make_response(
            jsonify({"error": f"{model_handle} model is not an AutoEncoder"}), 412
        )

    if "file" not in request.files:
        return make_response(jsonify({"error": "No file part"}), 412)

    file: FileStorage = request.files["file"]

    if file.filename == "":
        return make_response(jsonify({"error": "No file selected"}), 417)

    if allowed_file(file.filename):
        image: Image = file2image(file)
        autoencoder = get_autoencoder(model_handle)
        output: Image
        latent_z: np.ndarray
        output, latent_z = autoencoder(image)

        # convert it to b64 bytes
        b64_image = image2b64(output)
        return make_response(
            jsonify(dict(recon_image=b64_image, latent_z=latent_z.tolist())), 200
        )

    else:
        return make_response(jsonify({"error": f"{file.mimetype} not allowed"}), 412)


@app.route("/text/<model_handle>", methods=["POST"])
@cross_origin()
def text_api(model_handle="conv-sentimental-mclass") -> Response:
    if model_handle not in MODEL_REGISTER:
        return make_response(
            jsonify({"error": f"{model_handle} not found in registered models"}), 404
        )

    if "input_text" not in request.form:
        return make_response(
            jsonify({"error": "input_text not found in the form"}), 412
        )

    input_text = request.form["input_text"]

    text_func = get_text_function(model_handle)
    output = text_func(input_text)

    return make_response(jsonify(output), 200)


@app.route("/text/translate/<source_ln>/<target_ln>", methods=["POST"])
@cross_origin()
def translate_text(source_ln="de", target_ln="en") -> Response:
    if "source_text" not in request.form:
        return make_response(
            jsonify({"error": "input_text not found in the form"}), 412
        )

    source_text = request.form["source_text"]

    if source_ln == "de" and target_ln == "en":
        translate_func = get_text_translate_function("annotated-encoder-decoder-de-en")
        output = translate_func(source_text)

        return make_response(jsonify(output), 200)
    else:
        return make_response(
            jsonify({"error": f"{source_ln} -> {target_ln} not supported"}), 404
        )


def form_file_check(file_key):
    """
    Checks if the file key is present in request.files

    Args:
        file_key: the key used to retrieve file from the request.files dict
    """

    def decorator(api_func):
        @wraps(api_func)
        def wrapper(*args, **kwargs):
            if file_key not in request.files:
                return make_response(jsonify({"error": "No file part"}), 412)

            return api_func(*args, **kwargs)

        return wrapper

    return decorator


def model_handle_check(model_type):
    """
    Checks for the model_type and model_handle on the api function,
    model_type is a argument to this decorator, it steals model_handle and checks if it is
    present in the MODEL_REGISTER

    the api must have model_handle in it

    Args:
        model_type: the "type" of the model, as specified in the MODEL_REGISTER

    Returns:
        wrapped api function

    """

    def decorator(api_func):
        @wraps(api_func)
        def wrapper(*args, model_handle, **kwargs):
            if model_handle not in MODEL_REGISTER:
                return make_response(
                    jsonify(
                        {"error": f"{model_handle} not found in registered models"}
                    ),
                    404,
                )

            if (
                model_handle in MODEL_REGISTER
                and MODEL_REGISTER[model_handle]["type"] != model_type
            ):
                return make_response(
                    jsonify({"error": f"{model_handle} model is not an {model_type}"}),
                    412,
                )

            return api_func(*args, model_handle=model_handle, **kwargs)

        return wrapper

    return decorator


def get_image_from_request(
    from_request: Request, file_key: str
) -> Union[Response, Image]:
    file: FileStorage = from_request.files[file_key]

    if file.filename == "":
        return make_response(jsonify({"error": "No file selected"}), 417)

    if allowed_file(file.filename):
        image: Image = file2image(file)
        return image

    else:
        return make_response(jsonify({"error": f"{file.mimetype} not allowed"}), 412)


@app.route("/style-transfer/<model_handle>/<style_name>", methods=["POST"])
@cross_origin()
@model_handle_check(model_type="style-transfer")
def style_transfer_api(
    model_handle="fast-style-transfer", style_name="candy"
) -> Response:
    # check if its a valid style
    if style_name not in MODEL_REGISTER[model_handle]["model_stack"]:
        return make_response(
            jsonify({"error": f"{style_name} not in model_stack of {model_handle}"}),
            404,
        )

    # get the input image from the request
    returned_val: Union[Response, Image] = get_image_from_request(
        from_request=request, file_key="file"
    )

    # if a response is already created during process i.e. an error, then return that
    if isinstance(returned_val, Response):
        response: Response = returned_val
        return response

    image: Image = returned_val

    # now process the image
    style_transfer = get_style_transfer_function(model_handle, style_name)
    output: Image = style_transfer(image)

    # convert it to b64 bytes
    b64_image = image2b64(output)
    return make_response(jsonify(b64_image), 200)


@app.route("/image-captioning/<model_handle>", methods=["POST"])
@cross_origin()
@model_handle_check(model_type="image-caption")
@form_file_check(file_key="file")
def image_caption_api(model_handle="flickr8k-image-caption") -> Response:

    # get the input image from the request
    returned_val: Union[Response, Image] = get_image_from_request(
        from_request=request, file_key="file"
    )

    # if a response is already created during process i.e. an error, then return that
    if isinstance(returned_val, Response):
        response: Response = returned_val
        return response

    image: Image = returned_val

    # now process the image
    image_caption: Callable[[Image], str] = get_image_captioning_function(model_handle)
    output: str = image_caption(image)

    return make_response(jsonify({"caption": output}), 200)


@app.route("/human-pose", methods=["POST"])
@cross_origin()
def get_human_pose() -> Response:
    """

    Handles the human pose POST request, takes the pose image and identifies the human pose keypoints,
        stitches them together and returns a response with the image as b64 encoded, with the detected human pose

    Returns:
        (Response): b64 image string with the detected human pose

    """
    from models import get_pose

    if "file" not in request.files:
        return Response({"error": "No file part"}, status=412)

    file: FileStorage = request.files["file"]

    if file.filename == "":
        return Response({"error": "No file selected"}, status=417)

    if allowed_file(file.filename):
        image: Image = file2image(file)
        pose_img = get_pose(image)

        # convert it to b64 bytes
        b64_pose = image2b64(pose_img)

        return jsonify(b64_pose), 200

    else:
        return Response({"error": f"{file.mimetype} not allowed"}, status=412)
