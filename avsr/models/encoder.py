import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from avsr.models.conformer.conformer_backend import Conformer_back
from avsr.models.resnet.resnet import Resnet1D_front, Resnet2D_front


class FusionConformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_n_layer : int, 
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int, 
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.audio_model = AudioConformerEncoder(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        self.visual_model = VisualConformerEncoder(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        self.MLP = nn.Sequential(
            nn.Linear(encoder_d_model*2, encoder_d_model*4),
            nn.BatchNorm1d(encoder_d_model*4),
            nn.ReLU(),
            nn.Linear(encoder_d_model*4, encoder_d_model)
        )
        
    def forward(self, 
                video_inputs, video_input_lengths,
                audio_inputs, audio_input_lengths,
                *args, **kwargs):
        audio_feature = self.audio_model(audio_inputs, audio_input_lengths)
        visual_feature = self.visual_model(video_inputs, video_input_lengths)
        features = self.fusion(visual_feature, audio_feature)
        batch_seq_size = features.shape[:2]
        features = torch.flatten(features, end_dim=1)
        outputs = self.MLP(features)
        outputs = outputs.view(*batch_seq_size, -1)                
        return outputs
        
    def fusion(self, video_feature, audio_feature):
        '''
        ::ALERT:: This codes are only for pm video feature
        pm features losed 4 frames from original sequence because it is not padded.
        This code makes visual pm feature "zero padded" with front 2 frames & back 2 frames (& back extra 1 frame for sync)
        '''
        front_margin = 2
        back_margin = 3 if video_feature.size(-1) == (audio_feature.size(-1) + 3) else 2
        visual_feature = F.pad(visual_feature, (front_margin, back_margin), 'constant', 0)
        features = torch.cat([visual_feature, audio_feature], dim=-1)
        return features


class AudioConformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_n_layer : int, 
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int, 
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.front = Resnet1D_front(1, encoder_d_model)
        self.back  = Conformer_back(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        
    def forward(
        self, 
        video_inputs, video_input_lengths,
        audio_inputs, audio_input_lengths,
        *args, **kwargs
    ):
        outputs = self.front(audio_inputs)
        outputs = outputs.permute(0,2,1)
        outputs = self.back(outputs, audio_input_lengths)
        return outputs


class VisualConformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_n_layer : int, 
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int, 
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.front = Resnet2D_front(3, encoder_d_model)
        self.back  = Conformer_back(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        
    def forward(
        self, 
        video_inputs, video_input_lengths,
        audio_inputs, audio_input_lengths,
        *args, **kwargs
    ):
        #outputs = self.front(inputs)
        outputs = video_inputs
        print(f'visual_size : {outputs.size()}')
        outputs = self.back(outputs, video_input_lengths)
        return outputs