import torch
import torch.nn as nn

from models.model import *
from models.encoder import *
from models.decoder import *


def build_model(
    architecture='default' : str,
    encoder_d_model=None,
    encoder_n_head=None, 
    encoder_ff_dim=None, 
    encoder_dropout_p=None,
    decoder_d_model=None,
    decoder_n_head=None, 
    decoder_ff_dim=None, 
    decoder_dropout_p=None,
):
    if architecture == 'default':
        model = AttentionModel(*args, **kwargs)
        model.encoder = FusionConformerEncoder(
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        model.attdecoder = TransformerDecoder(
            decoder_d_model=decoder_d_model,
            decoder_n_head=decoder_n_head, 
            decoder_ff_dim=decoder_ff_dim, 
            decoder_dropout_p=decoder_dropout_p
        )
    return model