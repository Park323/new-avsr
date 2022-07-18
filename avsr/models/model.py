import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        pad_id : int,
    ):
        super().__init__()
        self.embedder = None
        self.encoder = None
        self.decoder = None
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        
    def forward(self,
                video_inputs,
                video_input_lengths,
                audio_inputs,
                audio_input_lengths,
                targets,
                target_lengths):
        pass


class HybridModel(EncoderDecoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                video_inputs,
                video_input_lengths,
                audio_inputs,
                audio_input_lengths,
                targets,
                target_lengths):
        features = self.encoder(video_inputs, video_input_lengths,
                                audio_inputs, audio_input_lengths)
        targets = F.one_hot(targets, num_classes = self.vocab_size)
        targets = self.embedder(targets.to(torch.float32))
        outputs = self.decoder(inputs=features, labels=targets, pad_id=self.pad_id)
        outputs = (F.log_softmax(outputs[0], dim=-1), F.log_softmax(outputs[1], dim=-1)) # (att_output, ctc_output)
        return outputs


class AttentionModel(EncoderDecoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                video_inputs,
                video_input_lengths,
                audio_inputs,
                audio_input_lengths,
                targets,
                target_lengths):
        features = self.encoder(video_inputs, video_input_lengths,
                                audio_inputs, audio_input_lengths)
        targets = F.one_hot(targets, num_classes = self.vocab_size)
        targets = self.embedder(targets.to(torch.float32))
        outputs = self.decoder(targets, features, pad_id=self.pad_id)
        outputs = F.log_softmax(outputs, dim=-1)
        return outputs


class CTCModel(EncoderDecoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                video_inputs,
                video_input_lengths,
                audio_inputs, 
                audio_input_lengths,
                *args, **kwargs):
        features = self.encoder(video_inputs, video_input_lengths,
                                audio_inputs, audio_input_lengths)
        outputs = self.decoder(features)
        outputs = F.log_softmax(outputs, dim=-1)
        return outputs
