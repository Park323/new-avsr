import torch
import torch.nn as nn
import pdb

"""
class Hybrid_Loss(nn.Module):
    '''
    Inputs : 
        outputs : tuple ( tensor (BxSxE), tensor (BxLxE) )
        targets : tensor (BxS)
    '''
    def __init__(
        self, 
        config
    ):
        super().__init__()
        self.ctc   = nn.CTCLoss(
            blank=config['unk_id'], 
            reduction='mean', 
            zero_infinity=True
        )
        self.att   = LabelSmoothingLoss(len(vocab), ignore_index=vocab.pad_id,
                                        smoothing=config.model.label_smoothing)
        
    def forward(self, outputs, targets, target_lengths):
        a = float(self.config.model.alpha)
        targets = targets
        att_out = outputs[0].contiguous().view(-1,outputs[0].shape[-1]) 
        ctc_out = outputs[1].contiguous().permute(1,0,2) # (B,L,E)->(L,B,E)
        att_loss = self.att(att_out, targets.contiguous().view(-1))
        ctc_loss = self.ctc(ctc_out, targets,
                            (torch.ones(ctc_out.shape[1])*ctc_out.shape[0]).to(torch.int),
                            target_lengths) 
        return a*att_loss + (1-a)*ctc_loss
        
        
class CTC_Loss(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.ctc   = nn.CTCLoss(blank=vocab.unk_id, reduction='mean', zero_infinity=True)
        
    def forward(self, outputs, targets, target_lengths):
        ctc_out = outputs.contiguous().permute(1,0,2) # (B,L,E)->(L,B,E)
        ctc_loss = self.ctc(ctc_out, targets,
                            (torch.ones(ctc_out.shape[1])*ctc_out.shape[0]).to(torch.int),
                            target_lengths)
        return ctc_loss
"""
        
        
class Attention_Loss(nn.Module):
    def __init__(
        self, 
        ignore_index,
        label_smoothing
    ):
        super().__init__()
        self.config = config
        self.att = nn.CrossEntropyLoss(
            size_average=True, 
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
    def forward(self, outputs, targets, *args, **kwargs):
        out = outputs.contiguous().view(-1,outputs.shape[-1])
        loss = self.att(out, targets.contiguous().view(-1))
        return loss

