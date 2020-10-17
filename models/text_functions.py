from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import Logger
from typing import List, Dict, Any

import en_core_web_sm
import torch
from spacy.lang.en import English
from torch import Tensor
from torch.jit import RecursiveScriptModule
from torchtext.vocab import Vocab
import torch.nn.functional as F

from utils import setup_logger

logger: Logger = setup_logger(__name__)


def classify_conv_sentimental_mclass(
    model: RecursiveScriptModule, vocabs: Dict[str, Vocab], sentence: str
) -> List[Dict[str, Any]]:
    """
    ConvNet for Text Classification, classifies a text question into HUM, NUM, LOC, ABBR

    Args:
        vocabs: the vocabulary dict that contains 'TEXT.vocab' and 'LABEL.vocab'
        model: the conv sentimental model, this was a custom conv net for multi class text classification
        sentence: the text who class is to be determined
    """

    nlp: English = en_core_web_sm.load()

    text_vocab: Vocab = vocabs["TEXT.vocab"]
    label_vocab: Vocab = vocabs["LABEL.vocab"]

    tokenized: List[str] = [tok.text for tok in nlp.tokenizer(sentence)]
    # if the length of sentence is less than 4 then pad it
    if len(tokenized) < 4:
        tokenized += ["<pad>"] * (4 - len(tokenized))

    indexed: List[int] = [text_vocab.stoi[t] for t in tokenized]
    input_tensor: Tensor = torch.LongTensor(indexed)
    input_tensor = input_tensor.unsqueeze(1)

    with torch.no_grad():
        predicted: Tensor = model(input_tensor)
        predicted = predicted.squeeze(0)
        predicted = F.softmax(predicted)
    sorted_values = predicted.argsort(descending=True).cpu().numpy()

    toppreds: List[Dict[str, Any]] = list(
        map(
            lambda x: {
                "label_idx": x.item(),
                "label_name": label_vocab.itos[x],
                "confidence": predicted[x].item(),
            },
            sorted_values,
        )
    )

    return toppreds
