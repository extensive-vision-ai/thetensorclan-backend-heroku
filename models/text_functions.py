from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from logging import Logger
from typing import List, Dict, Any

import en_core_web_sm
import de_core_news_sm

from spacy.lang.en import English
from spacy.lang.de import German
from torchvision.transforms import Compose

from .translator_models import annotated_encoder_decoder_de_en

import torch
from torch import Tensor
from torch.jit import RecursiveScriptModule
from torchtext.vocab import Vocab
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image

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


def flickr8k_image_captioning(
    encoder: RecursiveScriptModule,
    decoder: RecursiveScriptModule,
    wordmap: Dict[str, int],
    image: Image.Image,
    beam_size: int = 3,
) -> str:
    """

    Args:
        encoder: The Encoder Model
        decoder: The Decoder Model
        wordmap: The Word -> Index Map
        image: Input Image
        beam_size: Number of sequences to consider at each decode-step
    """
    rev_word_map: Dict[int, str] = {v: k for k, v in wordmap.items()}

    k: int = beam_size
    vocab_size: int = len(wordmap)

    trans: Compose = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image: Tensor = trans(image)  # (3, 256, 256)

    # Encode
    image: Tensor = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out: Tensor = encoder(
        image
    )  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size: int = encoder_out.size(1)
    encoder_dim: int = encoder_out.size(3)

    # Flatten encoding
    encoder_out: Tensor = encoder_out.view(
        1, -1, encoder_dim
    )  # (1, num_pixels, encoder_dim)
    num_pixels: int = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(
        k, num_pixels, encoder_dim
    )  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words: Tensor = torch.tensor(
        [[wordmap["<start>"]]] * k, dtype=torch.long
    )  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs: Tensor = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores: Tensor = torch.zeros(k, 1)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha: Tensor = torch.ones(
        k, 1, enc_image_size, enc_image_size
    )  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs: List = list()
    # complete_seqs_alpha: List = list()
    complete_seqs_scores: List = list()

    # Start decoding
    step: int = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings: Tensor = decoder.embedding(k_prev_words).squeeze(
            1
        )  # (s, embed_dim)

        awe, alpha = decoder.attention(
            encoder_out, h
        )  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(
            -1, enc_image_size, enc_image_size
        )  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), (h, c)
        )  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        # import pdb; pdb.set_trace()
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
        )  # (s, step+1)
        seqs_alpha = torch.cat(
            [seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1
        )  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [
            ind
            for ind, next_word in enumerate(next_word_inds)
            if next_word != wordmap["<end>"]
        ]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq: List[int] = complete_seqs[i]
    # alphas = complete_seqs_alpha[i]

    words: str = " ".join(
        [
            rev_word_map[ind]
            for ind in seq
            if ind not in {wordmap["<start>"], wordmap["<end>"], wordmap["<pad>"]}
        ]
    )

    return words
