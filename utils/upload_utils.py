from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Set
import io, base64

import torchaudio
import ffmpeg
from PIL import Image
from werkzeug.datastructures import FileStorage

from torch import Tensor
import soundfile as sf
import torch
import numpy as np

from utils import setup_logger

ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "wav", "webm"}

logger = setup_logger(__name__)


def allowed_file(filename: str) -> bool:
    """
    allowed_file

    Args:
        filename: the filename

    Returns:
        (bool): whether the file extension is permitted
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file2audiotensor(file: FileStorage) -> Tensor:
    """
    Reads the FileStorage object into Bytes, then converts it to a normalized torch tensor

    Args:
        file: the FileStorage object from flask Request

    Returns:
        (Tensor): the normalized tensor representation of the audio

    """
    logger.info(f"Got file {file.filename} of {file.mimetype}")
    file.stream.seek(0)
    byte_stream = io.BytesIO(file.read())
    logger.info(f"File Size: {byte_stream.getbuffer().nbytes}")
    file.close()

    # convert webm to wav
    process = (
        ffmpeg
        .input('pipe:0')
        .output('pipe:1', format='wav')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
    )

    output, err = process.communicate(input=byte_stream.read())

    riff_chunk_size = len(output) - 8
    # Break up the chunk size into four bytes, held in b.
    q = riff_chunk_size
    b = []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)

    # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
    riff = output[:4] + bytes(b) + output[8:]

    data, _ = sf.read(io.BytesIO(riff))
    waveform: Tensor = torch.from_numpy(data).float()

    return waveform.unsqueeze(0)


def file2image(file: FileStorage) -> Image.Image:
    """

    Reads the Bytes from FileStorage object and creates Image object from it

    Args:
        file (FileStorage): a flask file object received from the request

    Returns:
        (PIL.Image.Image)
    """
    logger.info(f"Got file {file.filename} of {file.mimetype}")
    file.stream.seek(0)
    byte_stream = io.BytesIO(file.read())
    logger.info(f"File Size: {byte_stream.getbuffer().nbytes}")
    file.close()
    image: Image.Image = Image.open(byte_stream)
    image: Image.Image = image.convert("RGB")

    return image


def image2b64(image: Image.Image) -> str:
    byte_image = io.BytesIO()
    image.save(byte_image, format="JPEG")
    img_str = base64.b64encode(byte_image.getvalue())
    img_base64 = bytes("data:image/jpeg;base64,", encoding="utf-8") + img_str

    return img_base64.decode("utf-8")
