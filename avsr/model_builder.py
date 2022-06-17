import torch
import torch.nn as nn

from avsr.models.model import *
from avsr.models.encoder import *
from avsr.models.decoder import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    vocab_size : int,
    pad_id : int,
    architecture:str = 'default',
    encoder_n_layer=None, 
    encoder_d_model=None,
    encoder_n_head=None, 
    encoder_ff_dim=None, 
    encoder_dropout_p=None,
    decoder_n_layer=None,
    decoder_d_model=None,
    decoder_n_head=None, 
    decoder_ff_dim=None, 
    decoder_dropout_p=None,
):
    if architecture=='default': architecture = 'hybrid'
    
    if architecture == 'hybrid':
        model = HybridModel(vocab_size=vocab_size, pad_id=pad_id)
        model.embedder = nn.Linear(vocab_size, decoder_d_model)
        model.encoder = FusionConformerEncoder(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        model.decoder = HybridDecoder(
            vocab_size = vocab_size,
            decoder_n_layer=decoder_n_layer, 
            decoder_d_model=decoder_d_model,
            decoder_n_head=decoder_n_head, 
            decoder_ff_dim=decoder_ff_dim, 
            decoder_dropout_p=decoder_dropout_p
        )
    elif architecture == 'att' or architecture=='attention':
        model = AttentionModel(vocab_size=vocab_size, pad_id=pad_id)
        model.embedder = nn.Linear(vocab_size, decoder_d_model)
        model.encoder = FusionConformerEncoder(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        model.decoder = TransformerDecoder(
            vocab_size = vocab_size,
            decoder_n_layer=decoder_n_layer, 
            decoder_d_model=decoder_d_model,
            decoder_n_head=decoder_n_head, 
            decoder_ff_dim=decoder_ff_dim, 
            decoder_dropout_p=decoder_dropout_p
        )
    elif architecture == 'ctc':
        model = CTCModel(vocab_size=vocab_size, pad_id=pad_id)
        model.embedder = nn.Linear(vocab_size, decoder_d_model)
        model.encoder = FusionConformerEncoder(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        model.decoder = LinearDecoder(
            vocab_size = vocab_size, 
            decoder_d_model=decoder_d_model,
        )

    print(f"# of total parameters : {count_parameters(model)}")

    return model