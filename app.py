from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
from typing import Tuple, Any

import numpy as np
from PIL import ImageFile
from PIL.Image import Image
from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
from werkzeug.datastructures import FileStorage

from models import get_classifier
from models.model_handlers import MODEL_REGISTER, get_generator, get_autoencoder
from utils import setup_logger, allowed_file, file2image
from utils.upload_utils import image2b64

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = setup_logger(__name__)
logger.info('=> Finished Importing')

# attach our logger to the system exceptions
sys.excepthook = lambda type, val, tb: logger.error("Unhandled exception:", exc_info=val)

app: Flask = Flask(__name__)
cors: CORS = CORS(app=app)
app.config['CORS_HEADERS'] = 'Content-Type'

if 'PRODUCTION' not in os.environ:
    app.config['DEBUG'] = True


@app.route("/")
@cross_origin()
def hello_thetensorclan() -> Tuple[Any, int]:
    return jsonify({'message': 'You\'ve reached the TensorClan Heroku Backend EndPoint'}), 200


@app.route("/classify/<model_handle>", methods=['POST'])
@cross_origin()
def classify_image_api(model_handle='resnet34-imagenet') -> Response:
    """

    Args:
        model_handle: the model handle string, should be in `models.model_handler.MODEL_REGISTER`

    Returns:
        (Response): if error then a json of {'error': 'message'} is sent
                    else return a json of sorted List[Dict[{'class_idx': idx, 'class_name': cn, 'confidence': 'c'}]]
    """
    if model_handle not in MODEL_REGISTER:
        return Response({'error': f'{model_handle} not found in registered models'}, status=404)

    if 'file' not in request.files:
        return Response({'error': 'No file part'}, status=412)

    file: FileStorage = request.files['file']

    if file.filename == '':
        return Response({'error': 'No file selected'}, status=417)

    if allowed_file(file.filename):
        image: Image = file2image(file)
        classifier = get_classifier(model_handle)
        output = classifier(image)
        return Response(json.dumps(output), status=200)

    else:
        return Response({'error': f'{file.mimetype} not allowed'}, status=412)


@app.route("/generators/<model_handle>", methods=['POST'])
@cross_origin()
def generator_api(model_handle='red-car-gan-generator') -> Response:
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
        return Response({'error': f'{model_handle} not found in registered models'}, status=404)

    if model_handle in MODEL_REGISTER and MODEL_REGISTER[model_handle]['type'] != 'gan-generator':
        return Response({'error': f'{model_handle} model is not a GAN'}, status=412)

    if 'latent_z' not in request.form:
        return Response({'error': 'latent_z not found in the form'}, status=412)

    latent_z = json.loads(f"[{request.form['latent_z']}]")
    latent_z = np.array(latent_z, dtype=np.float32)

    generator = get_generator(model_handle)
    output = generator(latent_z)

    # convert it to b64 bytes
    b64_image = image2b64(output)

    return jsonify(b64_image), 200


@app.route("/autoencoders/<model_handle>", methods=['POST'])
@cross_origin()
def autoencoder_api(model_handle='red-car-auto-encoder') -> Response:
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
        return Response({'error': f'{model_handle} not found in registered models'}, status=404)

    if model_handle in MODEL_REGISTER and MODEL_REGISTER[model_handle]['type'] != 'variational-auto-encoder':
        return Response({'error': f'{model_handle} model is not an AutoEncoder'}, status=412)

    if 'file' not in request.form:
        return Response({'error': 'No file part'}, status=412)

    file: FileStorage = request.files['file']

    if file.filename == '':
        return Response({'error': 'No file selected'}, status=417)

    if allowed_file(file.filename):
        image: Image = file2image(file)
        autoencoder = get_autoencoder(model_handle)
        output, latent_z = autoencoder(image)

        # convert it to b64 bytes
        b64_image = image2b64(output)
        return Response(json.dumps(dict(
            recon_image=b64_image,
            latent_z=latent_z
        )), status=200)

    else:
        return Response({'error': f'{file.mimetype} not allowed'}, status=412)


@app.route("/human-pose", methods=['POST'])
@cross_origin()
def get_human_pose() -> Response:
    """

    Handles the human pose POST request, takes the pose image and identifies the human pose keypoints,
        stitches them together and returns a response with the image as b64 encoded, with the detected human pose

    Returns:
        (Response): b64 image string with the detected human pose

    """
    from models import get_pose

    if 'file' not in request.files:
        return Response({'error': 'No file part'}, status=412)

    file: FileStorage = request.files['file']

    if file.filename == '':
        return Response({'error': 'No file selected'}, status=417)

    if allowed_file(file.filename):
        image: Image = file2image(file)
        pose_img = get_pose(image)

        # convert it to b64 bytes
        b64_pose = image2b64(pose_img)

        return jsonify(b64_pose), 200

    else:
        return Response({'error': f'{file.mimetype} not allowed'}, status=412)
