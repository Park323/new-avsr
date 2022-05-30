import torch
import torch.nn as nn


class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        pad_id : int,
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int,
        encoder_dropout_p : float,
        decoder_d_model : int, 
        decoder_n_head : int, 
        decoder_ff_dim : int, 
        decoder_dropout_p : float,
    ):
        super().__init__()
        self.embedder = None
        self.encoder = None
        self.attdecoder = None
        self.ctcdecoder = None
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
        targets = self.target_embedding(targets.to(torch.float32))
        output_a = self.decoder(targets, features, pad_id=self.pad_id)
        output_a = F.log_softmax(self.attdecoder(output_a), dim=-1)
        output_b = F.log_softmax(self.ctcdecoder(features), dim=-1)
        return (output_a, output_b)


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
        targets = self.target_embedding(targets.to(torch.float32))
        output = self.decoder(targets, features, pad_id=self.pad_id)
        output = F.log_softmax(self.attdecoder(output_a), dim=-1)
        return output


class CTCModel(EncoderDecoderModel):
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
        targets = self.target_embedding(targets.to(torch.float32))
        output = F.log_softmax(self.ctcdecoder(features), dim=-1)
        return output
