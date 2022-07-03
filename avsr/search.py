import pdb
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from avsr.models.model import *
from avsr.models.encoder import *
from avsr.models.decoder import *

EPSILON = 1e-200


def hybridSearch(
    model,
    video_inputs,
    video_input_lengths,
    audio_inputs,
    audio_input_lengths,
    max_len : int = 150,
    vocab_size : int = 1159,
    pad_id : int = 0,
    sos_id : int = 1,
    eos_id : int = 2,
    blank_id : int = 3,
    ctc_rate : float = 0.3,
    device = 'cpu',
    *args, **kwargs
):
    assert ctc_rate > 0, "ctc rate shoud be positive float number!"
    batch_size = video_inputs.size(0)
    features = model.encoder(video_inputs, video_input_lengths,
                             audio_inputs, audio_input_lengths)
    ctc_output = F.softmax(model.decoder.ctcdecoder(inputs=features), dim=-1)
    T = ctc_output.size(1)
    ctc_output = F.pad(ctc_output[0], (0,0,1,0))
    
    # predicted total sequence
    y_hats = torch.full((1,max_len), pad_id).to(device)
    y_hats[0,0] = sos_id
    
    # initialize score_dictionary for dynamic ctc score calculation
    y_n = [] ; y_b = [{(sos_id,):1}]
    for t in range(T+1):
        y_n.append({(sos_id,):0})
        y_b.append({(sos_id,):y_b[t-1][(sos_id,)] * ctc_output[t,blank_id].item()})
    
    # search...
    last_score = 0
    for idx in range(1, max_len):
        y_emb = model.embedder(F.one_hot(y_hats[:,:idx], num_classes=model.vocab_size).to(torch.float32))
        att_scores = F.log_softmax(model.decoder.attdecoder(inputs=features, labels=y_emb, pad_id=pad_id), dim=-1)
        att_scores = att_scores[0].cpu().numpy()
        
        ctc_scores = np.full((vocab_size), -np.inf)
        print(f"For index {idx}")
        for token_id in range(vocab_size):
            if token_id == sos_id:
                continue
            if token_id == pad_id:
                continue
            # get predicted tokens (B, 1)
            a_ctc = ctc_label_score(y_hats[0,:idx].cpu().numpy(), token_id, 
                                    X = ctc_output.cpu().numpy(), 
                                    T = T,
                                    y_n = y_n, 
                                    y_b = y_b,
                                    max_len=max_len, 
                                    sos_id=sos_id, 
                                    eos_id=eos_id, 
                                    blank_id=blank_id,)
                                    #last_score=last_score)
            ctc_scores[token_id] = a_ctc
            print(f"for token {token_id}, CTC : {a_ctc}, ATT : {att_scores[-1][token_id]}")
        scores = ctc_rate * ctc_scores + (1-ctc_rate) * att_scores[-1] 
        y_hat = np.argmax(scores)
        
        if y_hat == eos_id:
            break
        y_hats[0,idx] = y_hat
        
        ## score vanishing because of the large vocab size
        ## so revise it by multiplying last score
        #last_score += ctc_scores[y_hat]
    
    return y_hats[:,1:idx]


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
    pad_id : int = 0,
    sos_id : int = 1,
    eos_id : int = 2,
    blank_id : int = 3,
    device = 'cpu',
    *args, **kwargs
):
    batch_size = video_inputs.size(0)
    features = model.encoder(video_inputs, video_input_lengths,
                             audio_inputs, audio_input_lengths)
    ctc_output = F.softmax(model.decoder(inputs=features), dim=-1)
    T = ctc_output.size(1)
    ctc_output = F.pad(ctc_output[0], (0,0,1,0))
    
    # predicted total sequence
    y_hats = torch.full((1,max_len), pad_id).to(device)
    y_hats[0,0] = sos_id
    
    # initialize score_dictionary for dynamic ctc score calculation
    y_n = [] ; y_b = [{(sos_id,):1}]
    for t in range(T+1):
        y_n.append({(sos_id,):0})
        y_b.append({(sos_id,):y_b[t-1][(sos_id,)] * ctc_output[t,blank_id].item()})
    
    # search...
    last_score = 0
    for idx in range(1, max_len):
        y_emb = model.embedder(F.one_hot(y_hats[:,:idx], num_classes=model.vocab_size).to(torch.float32))
        
        scores = np.full((vocab_size), -np.inf)
        for token_id in range(vocab_size):
            if token_id == sos_id:
                continue
            if token_id == pad_id:
                continue
            # get predicted tokens (B, 1)
            score = ctc_label_score(y_hats[0,:idx].cpu().numpy(), token_id, 
                                    X = ctc_output.cpu().numpy(), 
                                    T = T,
                                    y_n = y_n, 
                                    y_b = y_b,
                                    max_len=max_len, 
                                    sos_id=sos_id, 
                                    eos_id=eos_id, 
                                    blank_id=blank_id,
                                    last_score=last_score)
            scores[token_id] = score
            
        y_hat = np.argmax(scores)
        
        if y_hat == eos_id:
            break
        y_hats[0,idx] = y_hat
        
        # score vanishing because of the large vocab size
        # so revise it by multiplying last score
        last_score += scores[y_hat]
    
    return y_hats[:,1:idx]


def ctc_label_score(g, c, X, T, y_n, y_b, max_len=150, sos_id=None, eos_id=None, blank_id=None, last_score=None):
    g = tuple(g)
    h = tuple([*g, c])
    if c == eos_id:
        score = y_n[T][g] + y_b[T][g]
        ## score vanishing because of the large vocab size
        ## so revise it by multiplying last score
        #if last_score:
        #    score /= (np.exp(last_score) + EPSILON)
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
        #if last_score:
        #    score /= (np.exp(last_score) + EPSILON)
        return np.log(score + EPSILON)
