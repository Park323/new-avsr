import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torchaudio
from torchaudio import ctc_decoder

from avsr.models.model import *
from avsr.models.encoder import *
from avsr.models.decoder import *


def hybridSearch(
    model,
    video_inputs,
    video_input_lengths,
    audio_inputs,
    audio_input_lengths,
    targets,
    target_lengths,
    max_len : int = 150,
    sos_id : int = 1,
    eos_id : int = 2,
    ctc_rate : float = 0.2,
    *args, **kwargs
):
    batch_size = video_inputs.size(0)
    features = model.encoder(video_inputs, video_input_lengths,
                             audio_inputs, audio_input_lengths)
    # total sequence will be stacked on y_hats
    y_hats = torch.tensor([[sos_id] for _ in range(batch_size)])
    # masking for ended sequence
    mask = torch.tensor([True for _ in range(batch_size)])
    
    # initialize score_dictionary for dynamic ctc score calculation
    y_n = [] ; y_b = []
    for t in range(max_len+1):
        y_n.append({sos_id:torch.full(X.size(0), 0)})
        y_b_prob = torch.full(X.size(0), 1)
        for u in range(t):
            y_b_prob *= y_b[u][sos_id] * X[:,u,:]
        y_b.append({sos_id:y_b_prob})
    y_n = [deepcopy(y_n) for _ in range(batch_size)]
    y_b = [deepcopy(y_b) for _ in range(batch_size)]
    
    while y_hats.size(1) <= max_len and mask.sum()!=0:
        
        y_emb = model.embedder(F.one_hot(y_hats, num_classes=model.vocab_size).to(torch.float32))
        att_scores = F.log_softmax(model.decoder.attdecoder(inputs=features, labels=y_emb, pad_id=pad_id), dim=-1)
        
        for token_id in range(vocab_size):
            if token_id == sos_id:
                continue
            if token_id == pad_id:
                continue
                
            y_hat = torch.tensor([[token_id] for _ in range(batch_size)])
            ctc_scores = F.softmax(model.decoder.ctcdecoder(inputs=features), dim=-1)
            
            # get predicted tokens (B, 1)
            a_ctc = torch.zeros((batch_size,1))
            a_ctc[mask] = ctc_label_score(y_hats[mask], y_hat[mask], ctc_scores[mask], y_n, y_b)
            a_att = att_scores[:,-1,:] #(B, E)
            scores[:,token_id] = ctc_rate * a_ctc + (1-ctc_rate) * a_att
        y_hat = torch.max(scores, dim=-1)
        # mask prediction
        y_hat[~mask] = pad_id
        # update mask
        mask[y_hat.view(-1)==eos_id] = False
        # stack prediction 
        y_hats = torch.hstack([y_hats, y_hat])
    return y_hats


def transformerSearch(
    model,
    video_inputs,
    video_input_lengths,
    audio_inputs,
    audio_input_lengths,
    targets,
    target_lengths,
    max_len : int = 150,
    pad_id : int = 0,
    sos_id : int = 1,
    eos_id : int = 2,
    *args, **kwargs
):
    batch_size = video_inputs.size(0)
    features = model.encoder(video_inputs, video_input_lengths,
                             audio_inputs, audio_input_lengths)
    # total sequence will be stacked on y_hats
    y_hats = torch.tensor([[sos_id] for _ in range(batch_size)])
    # masking for ended sequence. negative.
    mask = torch.tensor([False for _ in range(batch_size)])
    
    while y_hats.size(1) <= max_len and mask.sum()!=0:
        y_emb = model.embedder(F.one_hot(y_hats, num_classes=model.vocab_size).to(torch.float32))
        output = model.decoder(inputs=features, labels=y_emb, pad_id=pad_id)
        # get predicted tokens (B, 1)
        y_hat = torch.max(F.log_softmax(output, dim=-1), dim=-1)[:,-1:]
        # mask prediction
        y_hat[mask] = pad_id
        # update mask
        mask[y_hat.view(-1)==eos_id] = True
        # stack prediction 
        y_hats = torch.hstack([y_hats, y_hat])
    return y_hats


def ctc_label_score(g, c, X, y_n, y_b, max_len=150, eos_id=None, blank_id=None):
    h = torch.hstack([g,c])
    T = g.size(1) + 1
    score = torch.zeros(X.size(0))
    if y_n[T-1].get(g):
        mask = g[:,-1] == c
        score[mask] = y_n[T-1].get(g) + y_b[T-1].get(g)
        score[~mask] = y_b[T-1].get(g)
        
            
        y_b[T-1].get(g)
        
    if c==eos_id:
        return torch.log(y_n[T]+y_b[T])
    else:
        y_n[1][h] = 
        
    return