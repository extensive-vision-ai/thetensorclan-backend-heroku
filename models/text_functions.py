from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import Logger
from typing import List, Dict, Any

import en_core_web_sm
import de_core_news_sm

from spacy.lang.en import English
from spacy.lang.de import German

from .translator_models import annotated_encoder_decoder_de_en

import torch
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


def translate_annotated_encoder_decoder_de_en(
    model: annotated_encoder_decoder_de_en.EncoderDecoder,
    meta: Dict[str, Any],
    source_text: str,
) -> str:

    spacy_de: German = de_core_news_sm.load()

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    src_tok: List[str] = tokenize_de(source_text)

    src_idx: List[int] = [meta["SRC.vocab.stoi"][x] for x in src_tok] + [
        meta["SRC.vocab.stoi"][meta["EOS_TOKEN"]]
    ]
    src: Tensor = torch.LongTensor(src_idx)
    src_mask: Tensor = (src != meta["SRC.vocab.stoi"][meta["PAD_TOKEN"]]).unsqueeze(-2)
    src_length: Tensor = torch.tensor(len(src))

    # convert to batch size 1
    src = src.unsqueeze(0)
    src_mask = src_mask.unsqueeze(0)
    src_length = src_length.unsqueeze(0)

    output = annotated_encoder_decoder_de_en.greedy_decode(
        model,
        src,
        src_mask,
        src_length,
        max_len=100,
        sos_index=meta["TRG.vocab.stoi"][meta["SOS_TOKEN"]],
        eos_index=meta["TRG.vocab.stoi"][meta["EOS_TOKEN"]],
    )

    return " ".join([meta["TRG.vocab.itos"][x] for x in output])
