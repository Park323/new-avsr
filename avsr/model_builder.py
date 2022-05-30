import torch
import torch.nn as nn

from models.model import *
from models.encoder import *
from models.decoder import *


def build_model():
    model = AttentionModel()
    model.encoder = FusionConformerEncoder()
    model.attdecoder = TransformerDecoder()
    return model