from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List

from logging import Logger

import torch
import torchaudio.transforms as T
from torch import Tensor
from torch.jit import RecursiveScriptModule
from torchaudio.transforms import MelSpectrogram


def speech_recognition_residual_text(
    model: RecursiveScriptModule, input_audio: Tensor
) -> str:
    char_map_str = """
    ' 0
    <SPACE> 1
    a 2
    b 3
    c 4
    d 5
    e 6
    f 7
    g 8
    h 9
    i 10
    j 11
    k 12
    l 13
    m 14
    n 15
    o 16
    p 17
    q 18
    r 19
    s 20
    t 21
    u 22
    v 23
    w 24
    x 25
    y 26
    z 27
    """
    char_map: Dict[str, int] = {}
    index_map: Dict[int, str] = {}
    for line in char_map_str.strip().split("\n"):
        ch, index = line.split()
        char_map[ch] = int(index)
        index_map[int(index)] = ch
    index_map[1] = " "

    def text_to_int(text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == " ":
                char = char_map["<SPACE>"]
            else:
                char = char_map[c]
            int_sequence.append(char)
        return int_sequence

    def int_to_text(labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(index_map[i])
        return "".join(string).replace("<SPACE>", " ")

    audio_transform: MelSpectrogram = T.MelSpectrogram()

    audio_tensor: Tensor = audio_transform(input_audio)
    audio_tensor: Tensor = audio_tensor.unsqueeze(0)

    with torch.no_grad():
        output: Tensor = model(audio_tensor)

    arg_maxes = torch.argmax(output, dim=2)

    decoded: List[str] = []
    blank_label = 28
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decoded.append(int_to_text(decode))

    return decoded[0]
