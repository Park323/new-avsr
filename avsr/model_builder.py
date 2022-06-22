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
    architecture : str = 'default',
    loss_fn : str = 'default',
    front_dim=None,
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
    rank = 0,
):
    if architecture=='default': architecture = 'audio_visual'
    if loss_fn=='default': architecture = 'hybrid'
    
    if rank==0 or rank is None:
        print(f"Build {loss_fn} {architecture} model...")
    
    
    # Define Model
    if loss_fn == 'hybrid':
        model = HybridModel(vocab_size=vocab_size, pad_id=pad_id)
    elif loss_fn == 'att' or architecture=='attention':
        model = AttentionModel(vocab_size=vocab_size, pad_id=pad_id)
    elif loss_fn == 'ctc':
        model = CTCModel(vocab_size=vocab_size, pad_id=pad_id)

    # Define Embedder
    model.embedder = nn.Linear(vocab_size, decoder_d_model)

    # Define Encoder
    if architecture == 'audio_visual':
        model.encoder = FusionConformerEncoder(
            front_dim = front_dim,
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
    elif architecture == 'audio':
        model.encoder = AudioConformerEncoder(
            front_dim = front_dim,
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
    elif architecture == 'video':
        model.encoder = VisualConformerEncoder(
            front_dim = front_dim,
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )

    # Define Decoder
    if loss_fn == 'hybrid':
        model.decoder = HybridDecoder(
            vocab_size = vocab_size,
            decoder_n_layer=decoder_n_layer, 
            decoder_d_model=decoder_d_model,
            decoder_n_head=decoder_n_head, 
            decoder_ff_dim=decoder_ff_dim, 
            decoder_dropout_p=decoder_dropout_p
        )
    elif loss_fn == 'att' or architecture=='attention':
        model.decoder = TransformerDecoder(
            vocab_size = vocab_size,
            decoder_n_layer=decoder_n_layer, 
            decoder_d_model=decoder_d_model,
            decoder_n_head=decoder_n_head, 
            decoder_ff_dim=decoder_ff_dim, 
            decoder_dropout_p=decoder_dropout_p
        )
    elif loss_fn == 'ctc':
        model.decoder = LinearDecoder(
            vocab_size = vocab_size, 
            decoder_d_model=decoder_d_model,
        )
        
    if rank==0 or rank is None:
        print("Build complete.")
        print(f"# of total parameters : {count_parameters(model)}")

    return model