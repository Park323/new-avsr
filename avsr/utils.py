from torch.optim import Adam

from avsr.loss import *
from avsr.metric import Metric
from avsr.scheduler.noam import NoamLR
from avsr.scheduler.none import Unscheduler
from avsr.search import *

def get_criterion(
    loss_fn : str = 'default',
    ignore_index : int = None,
    label_smoothing : float = 0.0,
    blank_id : int = None,
):
    if loss_fn=='default': loss_fn = 'hybrid'
    
    if loss_fn=='hybrid':
        criterion = Hybrid_Loss(
            ignore_index = ignore_index,
            label_smoothing = label_smoothing,
            blank_id = blank_id,        
        )
    elif loss_fn=='att':
        criterion = Attention_Loss(
            ignore_index,
            label_smoothing
        )
    elif loss_fn=='ctc':
        criterion = CTC_Loss(
            blank_id,
        )
    return criterion

def get_metric(vocab, log_path, unit:str='character', error_type:str='cer'):
    if unit=='character' and error_type=='ger': return None
    return Metric(vocab, log_path, unit=unit, error_type=error_type)
    
def get_optimizer(
    params,
    learning_rate,
    epochs : int = None,
    warmup : int = None,
    steps_per_epoch : int = 0,
    init_lr : float = None,
    final_lr : float = None,
    scheduler : str = 'noam',
):
    optimizer = Adam(params, learning_rate)
    if scheduler=='noam':
        scheduler = NoamLR(
            optimizer,
            [warmup],
            [epochs],
            [steps_per_epoch], # dataset_size / batch_size (= len(dataloader))
            [init_lr],
            [learning_rate],
            [final_lr],
        ) 
    else:
        scheduler = Unscheduler(learning_rate)
    return optimizer, scheduler
    
def select_search(method='default'):
    if method=='default': method = 'hybrid'
    
    if method=='hybrid':
        return hybridSearch
    elif method=='att':
        return transformerSearch
    elif method=='ctc':
        return ctcSearch