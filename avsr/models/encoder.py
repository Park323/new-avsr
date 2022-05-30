import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from conformer_backend import Conformer_back
from resnet import Resnet1D_front, Resnet2D_front


class FusionConformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int, 
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.audio_model = AudioConformerEncoder(
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        self.visual_model = VisualConformerEncoder(
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        self.MLP = nn.Sequential(
            nn.Linear(encoder_d_model*2, encoder_d_model*4),
            nn.BatchNorm1d(encoder_.d_model*4),
            nn.ReLU(),
            nn.Linear(encoder_d_model*4, encoder_d_model)
        )
        
    def forward(self, 
                video_inputs, video_input_lengths,
                audio_inputs, audio_input_lengths,
                *args, **kwargs):
        audio_feature = audio_model(audio_inputs, audio_input_lengths)
        visual_feature = visual_model(video_inputs, video_input_lengths)
        features = torch.cat([visual_feature, audio_feature], dim=-1)
        batch_seq_size = features.shape[:2]
        features = torch.flatten(features, end_dim=1)
        outputs = self.MLP(features)
        outputs = outputs.view(*batch_seq_size, -1)                
        return outputs


class AudioConformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int, 
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.front = AudioFrontEnd(config)
        self.back  = Conformer_back(
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        
    def forward(
        self,
        audio_inputs, 
        audio_input_lengths,
        *args, **kwargs
    ):
        outputs = self.front(audio_inputs)
        outputs = outputs.permute(0,2,1)
        outputs = self.back(outputs, inputs.size(0))
        return outputs


class VisualConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.front = VisualFrontEnd(config)
        self.back  = Conformer_back(
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        
    def forward(
        self,
        video_inputs, 
        video_input_lengths,
    ):
        #outputs = self.front(inputs)
        outputs = inputs
        outputs = self.back(outputs, input_lengths)
        return outputs