from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
from typing import Tuple, Any

from PIL import ImageFile
from PIL.Image import Image
from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
from werkzeug.datastructures import FileStorage

from models import get_classifier
from models.classifier_model_handler import MODEL_REGISTER
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


@app.route("/human-pose", methods=['POST'])
@cross_origin()
def classify_image_api() -> Response:
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
