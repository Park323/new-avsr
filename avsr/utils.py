from torch.optim import Adam

from avsr.loss import *
from avsr.metric import Metric
from avsr.scheduler.noam import NoamLR


def get_criterion(
    ignore_index : int = None,
    label_smoothing : float = 0.0,
    blank_id : int = None,
):
#    criterion = Attention_Loss(
#        ignore_index,
#        label_smoothing
#    )
    criterion = Hybrid_Loss(
        ignore_index = ignore_index,
        label_smoothing = label_smoothing,
        blank_id = blank_id,        
    )
    return criterion

def get_metric(vocab, log_path):
    return Metric(vocab, log_path)
    
def get_optimizer(
    params,
    learning_rate,
    epochs : int = None,
    steps_per_epoch : int = 0,
):
    optimizer = Adam(params, learning_rate)
    scheduler = NoamLR(
        optimizer,
        [float(25000/steps_per_epoch)],
        [epochs],
        [steps_per_epoch], # dataset_size / batch_size (= len(dataloader))
        [1e-10],
        [learning_rate],
        [1e-10]
    ) 
    return optimizer, scheduler