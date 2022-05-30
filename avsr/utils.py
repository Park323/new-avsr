from torch.optim import Adam

from loss import *
from metric import Metric
from scheduler.noam import NoamLR


def get_criterion(
    ignore_index : int,
    label_smoothing : float = 0.0
):
    criterion = Attention_Loss(
        ignore_index,
        label_smoothing
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
        [2.5],
        [epochs],
        [steps_per_epoch], # dataset_size / batch_size (= len(dataloader))
        [1e-10],
        [learning_rate],
        [1e-10]
    ) 
    return optimizer, scheduler