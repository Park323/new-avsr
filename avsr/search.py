import pdb
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from avsr.models.model import *
from avsr.models.encoder import *
from avsr.models.decoder import *

EPSILON = 1e-200


def end_detect(hypothesis, length, M=12, threshold=16):
    # increasement in M -> Allow long decreasing
    # increasement in threshold -> Allow deeply decreased score
    if len(hypothesis[0])==5:
        _, max_score, _, _, _ = max(hypothesis, key=lambda x: x[1])
        _, max_l, _, _, _ = max(hypothesis[-M:], key=lambda x: x[1])
    else:
        _, max_score, _ = max(hypothesis, key=lambda x: x[1])
        _, max_l, _ = max(hypothesis[-M:], key=lambda x: x[1])
    if max_score - max_l >= threshold:
        return True
    return False


def hybridSearch(
    model,
    video_inputs,
    video_input_lengths,
    audio_inputs,
    audio_input_lengths,
    max_len : int = 150,
    vocab_size : int = 1159,
    beam_size : int = 1,
    pad_id : int = 0,
    sos_id : int = 1,
    eos_id : int = 2,
    blank_id : int = 3,
    ctc_rate : float = 0.3,
    device = 'cpu',
    *args, **kwargs
):
    assert ctc_rate > 0, "ctc rate shoud be positive float number!"
    
    disabled_token = [pad_id, sos_id]
    
    batch_size = video_inputs.size(0)
    features = model.encoder(video_inputs, video_input_lengths,
                             audio_inputs, audio_input_lengths)
    
    ctc_output = F.softmax(model.decoder.ctcdecoder(inputs=features), dim=-1)
    T = ctc_output.size(1)
    ctc_output = F.pad(ctc_output[0], (0,0,1,0))
    
    # initialize score_dictionary for dynamic ctc score calculation
    y_n = [] ; y_b = [{(sos_id,):1}]
    for t in range(T+1):
        y_n.append({(sos_id,):0})
        y_b.append({(sos_id,):y_b[t-1][(sos_id,)] * ctc_output[t,blank_id].item()})
    
    # sos sequence
    y_hats = torch.full((1,max_len), pad_id).to(device)
    y_hats[0,0] = sos_id
    
    # length-limited hypothesis, initialized 'hypo_0'
    hypo_l = [(y_hats,0,0,0,0)]
    
    # completed hypothesis
    hypothesis = []
    
    # search...
    for length in range(1, max_len):
        # complete sub hypothesis
        hypo_sub = []
    
        # make beam
        beam = []
        min_score = -float('inf')
        
        while hypo_l:
            g, g_score, g_len, g_att, g_ctc = hypo_l.pop(0)
            
            # calculate attention score
            y_emb = model.embedder(F.one_hot(g[:,:length], num_classes=model.vocab_size).to(torch.float32))
            att_scores = F.log_softmax(model.decoder.attdecoder(inputs=features, labels=y_emb, pad_id=pad_id), dim=-1)
            att_scores = att_scores[0].cpu().numpy() + g_att
            
            ctc_scores = ctc_label_scores(g[0,:length].cpu().numpy(), vocab_size, 
                                      X = ctc_output.cpu().numpy(), 
                                      T = T,
                                      y_n = y_n, 
                                      y_b = y_b,
                                      max_len=max_len, 
                                      sos_id=sos_id, 
                                      eos_id=eos_id, 
                                      pad_id=pad_id,
                                      blank_id=blank_id,)
            
            scores = ctc_rate * ctc_scores + (1-ctc_rate) * att_scores
            
            y_hat = g.detach().clone()
            y_hat[0, length] = eos_id
            hypo_sub.append((y_hat, scores[eos_id].item(), length))
            
            if len(beam)<beam_size or (scores[np.isin(scores,disabled_token)]>min_score).sum():
                for token in np.arange(vocab_size)[scores>min_score]:
                    if token in disabled_token:
                        continue
                    y_hat = y_hat.clone()
                    y_hat[0, length] = int(token)
                    beam.append((y_hat, scores[token].item(), length, att_scores[token].item(), ctc_scores[token].item()))
                beam = sorted(beam, key=lambda x: x[1], reverse=True) # Sort by score, descending
                if len(beam)>beam_size:
                    for _ in range(len(beam)-beam_size):
                        beam.pop(-1)
                        min_score = beam[-1][1]
        
        # Add complete hypothesis
        hypothesis.append(max(hypo_sub, key=lambda x: x[1]))
            
        #end detect
        if end_detect(hypothesis, length):
            break
        
        hypo_l = beam
        
    max_hypothesis, max_score, length, _, _ = max(hypothesis, key=lambda x: x[1])
    
    return max_hypothesis[:, 1:length]


def transformerSearch(
    model,
    video_inputs,
    video_input_lengths,
    audio_inputs,
    audio_input_lengths,
    max_len : int = 150,
    pad_id : int = 0,
    sos_id : int = 1,
    eos_id : int = 2,
    device = 'cpu',
    *args, **kwargs
):
    batch_size = video_inputs.size(0)
    features = model.encoder(video_inputs, video_input_lengths,
                             audio_inputs, audio_input_lengths)
    # total sequence
    y_hats = torch.full((1,max_len), pad_id).to(device)
    y_hats[0,0] = sos_id
    
    for idx in range(1, max_len):
        y_emb = model.embedder(F.one_hot(y_hats[:,:idx], num_classes=model.vocab_size).to(torch.float32))
        output = model.decoder(inputs=features, labels=y_emb, pad_id=pad_id)
        output = output[0] if isinstance(output, tuple) else output
        # get predicted token
        y_hat = torch.argmax(F.log_softmax(output, dim=-1), dim=-1)[0,-1]
        if y_hat==eos_id:
            break
        y_hats[0,idx] = y_hat
    return y_hats[:,1:idx]


def ctcSearch(
    model,
    video_inputs,
    video_input_lengths,
    audio_inputs,
    audio_input_lengths,
    max_len : int = 150,
    vocab_size : int = 1159,
    beam_size : int = 1,
    pad_id : int = 0,
    sos_id : int = 1,
    eos_id : int = 2,
    blank_id : int = 3,
    device = 'cpu',
    *args, **kwargs
):
    ctc_disabled_token = [pad_id, sos_id]

    batch_size = video_inputs.size(0)
    features = model.encoder(video_inputs, video_input_lengths,
                             audio_inputs, audio_input_lengths)
    
    ctc_output = F.softmax(model.decoder.ctcdecoder(inputs=features), dim=-1)
    T = ctc_output.size(1)
    ctc_output = F.pad(ctc_output[0], (0,0,1,0))
    
    # initialize score_dictionary for dynamic ctc score calculation
    y_n = [] ; y_b = [{(sos_id,):1}]
    for t in range(T+1):
        y_n.append({(sos_id,):0})
        y_b.append({(sos_id,):y_b[t-1][(sos_id,)] * ctc_output[t,blank_id].item()})
    
    # sos sequence
    y_hats = torch.full((1,max_len), pad_id).to(device)
    y_hats[0,0] = sos_id
    
    # length-limited hypothesis, initialized 'hypo_0'
    hypo_l = [(y_hats,0,0)]
    
    # completed hypothesis
    hypothesis = []
    
    # search...
    for length in range(1, max_len):
        # complete sub hypothesis
        hypo_sub = []
    
        # make beam
        beam = []
        min_score = -float('inf')
        
        while hypo_l:
            g, g_score, g_len = hypo_l.pop(0)
            
            scores = ctc_label_scores(g[0,:length].cpu().numpy(), vocab_size, 
                                      X = ctc_output.cpu().numpy(), 
                                      T = T,
                                      y_n = y_n, 
                                      y_b = y_b,
                                      max_len=max_len, 
                                      sos_id=sos_id, 
                                      eos_id=eos_id, 
                                      pad_id=pad_id,
                                      blank_id=blank_id,)
            
            y_hat = g.detach().clone()
            y_hat[0, length] = eos_id
            hypo_sub.append((y_hat, scores[eos_id].item(), length))
            
            if len(beam)<beam_size or (scores[np.isin(scores,ctc_disabled_token)]>min_score).sum():
                for token in np.arange(vocab_size)[scores>min_score]:
                    if token in ctc_disabled_token:
                        continue
                    y_hat = y_hat.clone()
                    y_hat[0, length] = int(token)
                    beam.append((y_hat, scores[token].item(), length))
                beam = sorted(beam, key=lambda x: x[1], reverse=True) # Sort by score, descending
                if len(beam)>beam_size:
                    for _ in range(len(beam)-beam_size):
                        beam.pop(-1)
                        min_score = beam[-1][1]
        
        # Add complete hypothesis
        hypothesis.append(max(hypo_sub, key=lambda x: x[1]))
        
        #end detect
        if end_detect(hypothesis, length):
            break
        
        hypo_l = beam
        
    max_hypothesis, max_score, length = max(hypothesis, key=lambda x: x[1])
    
    return max_hypothesis[:, 1:length]


def ctc_label_score(g, c, X, T, y_n, y_b, max_len=150, sos_id=None, eos_id=None, blank_id=None, last_score=None):
    g = tuple(g)
    h = tuple([*g, c])
    if c == eos_id:
        score = y_n[T][g] + y_b[T][g]
        return np.log(score + EPSILON)
    else:
        y_n[1][h] = X[1][c] if g==(sos_id,) else 0
        y_b[1][h] = 0
        psi = y_n[1][h]
        for t in range(2, T+1):
            phi = y_b[t-1][g] if g[-1]==c else y_b[t-1][g]+y_n[t-1][g]
            y_n[t][h] = (y_n[t-1][h] + phi)*X[t][c]
            y_b[t][h] = (y_b[t-1][h] + y_n[t-1][h])*X[t][blank_id]
            psi += phi*X[t][c]
        #pdb.set_trace()
        score = psi
        return np.log(score + EPSILON)


def ctc_label_scores(g, vocab_size, X, T, y_n, y_b, max_len=150, sos_id=None, eos_id=None, blank_id=None, pad_id=None, last_score=None):
    g = tuple(g)
    
    vocabs = [c for c in range(vocab_size) if c not in [sos_id, eos_id, pad_id]]
    
    yn = np.zeros(vocab_size)
    yn[:] = X[1] if g==(sos_id,) else 0
    yb = np.zeros(vocab_size)
    
    y_n[1].update({tuple([*g, c]):yn[c].item() for c in vocabs})
    y_b[1].update({tuple([*g, c]):0 for c in vocabs})
    
    psi = yn.copy()
    phi = np.zeros(vocab_size)
    
    for t in range(2, T+1):
        _yn = yn.copy()
        _yb = yb.copy()
        
        phi[:] = y_b[t-1][g]+y_n[t-1][g]
        phi[g[-1]] = y_b[t-1][g]
        
        yn = (_yn + phi)*X[t]
        y_n[t].update({tuple([*g, c]):yn[c].item() for c in vocabs})
        
        yb = (_yb + _yn)*X[t][blank_id]
        y_b[t].update({tuple([*g, c]):yb[c].item() for c in vocabs})
        
        psi += phi*X[t]
    
    psi[eos_id] = y_n[T][g] + y_b[T][g]
    scores = np.log(psi + EPSILON)
    return scores
