import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridDecoder(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        decoder_n_layer : int,
        decoder_d_model : int, 
        decoder_n_head : int, 
        decoder_ff_dim : int, 
        decoder_dropout_p : float,
    ):
        super().__init__()
        self.attdecoder = TransformerDecoder(
            vocab_size=vocab_size,
            decoder_n_layer=decoder_n_layer,
            decoder_d_model=decoder_d_model, 
            decoder_n_head=decoder_n_head, 
            decoder_ff_dim=decoder_ff_dim, 
            decoder_dropout_p=decoder_dropout_p,
        )
        self.ctcdecoder = LinearDecoder(vocab_size=vocab_size, decoder_d_model=decoder_d_model)
    
    def forward(self, labels, inputs, pad_id=None, **kwargs):
        output_a = self.attdecoder(inputs=inputs, labels=labels, pad_id=pad_id)
        output_b = self.ctcdecoder(inputs=inputs)
        return (output_a, output_b) # (B, M, D)


class TransformerDecoder(nn.Module):
    '''
    Inputs : (B x S x E), (B x T x E)
    '''
    def __init__(
        self,
        vocab_size : int,
        decoder_n_layer : int,
        decoder_d_model : int, 
        decoder_n_head : int, 
        decoder_ff_dim : int, 
        decoder_dropout_p : float,
    ):
        super().__init__()
        decoder = nn.TransformerDecoderLayer(decoder_d_model, decoder_n_head, 
                                             decoder_ff_dim, decoder_dropout_p)
        self.decoder = nn.TransformerDecoder(decoder, decoder_n_layer)
        self.fc = nn.Linear(decoder_d_model, vocab_size)
        
    def forward(self, labels, inputs, pad_id=None, **kwargs):
        label_mask = self.generate_square_subsequent_mask(labels.shape[1]).to(inputs.device)
        label_pad_mask = self.get_attn_pad_mask(torch.argmax(labels, dim=-1), pad_id) if pad_id else None
        
        labels = labels.permute(1,0,2)
        inputs = inputs.permute(1,0,2)
        
        outputs = self.decoder(labels, inputs, 
                               tgt_mask=label_mask,
                               tgt_key_padding_mask=label_pad_mask)
        
        outputs = outputs.permute(1,0,2)
        outputs = self.fc(outputs)
        return outputs
        
    def generate_square_subsequent_mask(self, sz):
        mask = torch.full((sz, sz),-float('inf'))
        mask = torch.triu(mask, diagonal=1)
        return mask
        
    def get_attn_pad_mask(self, seq, pad):
        batch_size, len_seq = seq.size()
        pad_attn_mask = seq.eq(pad)
        return pad_attn_mask
        
        
class LinearDecoder(nn.Module):
    '''
    Inputs : (B x S x E), (B x T x E)
    '''
    def __init__(
        self, 
        vocab_size : int,
        decoder_d_model : int, 
        *args,
        **kwargs
    ):
        super().__init__()
        self.fc = nn.Linear(decoder_d_model, vocab_size)
        
    def forward(self, inputs, pad_id=None, **kwargs):
        outputs = self.fc(inputs)
        return outputs
        