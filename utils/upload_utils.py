from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Set
import io, base64
from PIL import Image
from werkzeug.datastructures import FileStorage

from utils import setup_logger

ALLOWED_EXTENSIONS: Set[str] = {'png', 'jpg', 'jpeg'}

logger = setup_logger(__name__)


def allowed_file(filename: str) -> bool:
    """
    allowed_file

    Args:
        filename: the filename

    Returns:
        (bool): whether the file extension is permitted
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def file2image(file: FileStorage) -> Image.Image:
    """

    Args:
        file (FileStorage): a flask file object received from the request

    Returns:
        (PIL.Image.Image)
    """
    logger.info(f'Got file {file.filename} of {file.mimetype}')
    file.stream.seek(0)
    byte_stream = io.BytesIO(file.read())
    logger.info(f'File Size: {byte_stream.getbuffer().nbytes}')
    file.close()
    image: Image.Image = Image.open(byte_stream)
    image: Image.Image = image.convert('RGB')

    return image


def image2b64(image: Image.Image) -> str:
    byte_image = io.BytesIO()
    image.save(byte_image, format='JPEG')
    img_str = base64.b64encode(byte_image.getvalue())
    img_base64 = bytes('data:image/jpeg;base64,', encoding='utf-8') + img_str

    return img_base64.decode("utf-8")
