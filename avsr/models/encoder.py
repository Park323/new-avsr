import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from avsr.models.conformer.conformer_backend import Conformer_back
from avsr.models.resnet.resnet import Resnet1D_front, Resnet2D_front


class FusionConformerEncoder(nn.Module):
    def __init__(
        self,
        front_dim : int,
        encoder_n_layer : int, 
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int, 
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.audio_model = AudioConformerEncoder(
            front_dim = front_dim,
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        self.visual_model = VisualConformerEncoder(
            front_dim = front_dim,
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        self.MLP = MLPLayer(encoder_d_model * 2, encoder_d_model)
        
    def forward(self, 
                video_inputs, video_input_lengths,
                audio_inputs, audio_input_lengths,
                *args, **kwargs):
        audio_feature = self.audio_model(None, None, audio_inputs, audio_input_lengths)
        visual_feature = self.visual_model(video_inputs, video_input_lengths, None, None)
        features = self.fusion(visual_feature, audio_feature)
        batch_seq_size = features.shape[:2]
        outputs = self.MLP(features)
        outputs = outputs.view(*batch_seq_size, -1)                
        return outputs
        
    def fusion(self, visual_feature, audio_feature):
        '''
        ::ALERT:: This codes are only for pm video feature
        pm features losed 4 frames from original sequence because it is not padded.
        This code makes visual pm feature "zero padded" with front 2 frames & back 2 frames (& back extra 1 frame for sync)
        '''
        diff = audio_feature.size(1) - visual_feature.size(1)
#        if diff > 10:
#            print(f"feature size differs {diff} frames")
#            print()
        front_margin = diff//2
        back_margin = diff - front_margin
        visual_feature = F.pad(visual_feature, (0, 0, front_margin, back_margin), 'constant', 0)
        
        features = torch.cat([visual_feature, audio_feature], dim=-1)
        return features


class AudioConformerEncoder(nn.Module):
    def __init__(
        self,
        front_dim : int,
        encoder_n_layer : int, 
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int, 
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.front = Resnet1D_front(1, front_dim,)
        self.back  = Conformer_back(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        self.projection = nn.Linear(front_dim, encoder_d_model)
        
    def forward(
        self, 
        video_inputs, video_input_lengths,
        audio_inputs, audio_input_lengths,
        *args, **kwargs
    ):
        outputs = self.front(audio_inputs)
        outputs = outputs.permute(0,2,1) # (B, L, C)
        outputs = self.projection(outputs)
        outputs = self.back(outputs, audio_input_lengths)
        return outputs


class VisualConformerEncoder(nn.Module):
    def __init__(
        self,
        front_dim : int,
        encoder_n_layer : int, 
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int, 
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.front = Resnet2D_front(3, front_dim,)
        self.back  = Conformer_back(
            encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p
        )
        self.projection = nn.Linear(front_dim, encoder_d_model)
        
    def forward(
        self, 
        video_inputs, video_input_lengths,
        audio_inputs, audio_input_lengths,
        *args, **kwargs
    ):
        #outputs = self.front(inputs)
        outputs = video_inputs
        outputs = self.projection(outputs)
        outputs = self.back(outputs, video_input_lengths)
        return outputs
        
        
class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim*4)
        self.linear_2 = nn.Linear(output_dim*4, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.linear_1(x)
        output = output.permute((0,2,1))
        output = self.batchnorm(output)
        output = output.permute((0,2,1))
        output = self.relu(output)
        output = self.linear_2(output)
        return output