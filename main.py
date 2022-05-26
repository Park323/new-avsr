import os
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
        

def setup(rank, world_size):
    """
    world_size : number of processes
    rank : this should be a number between 0 and world_size-1
    """
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    '''
    Initialize the process group
    Rule of thumb choosing backend
    NCCL -> GPU training / Gloo -> CPU training
    check table here : https://pytorch.org/docs/stable/distributed.html
    '''
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    
def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, dataset):

    setup(rank, world_size)
    
    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    outputs = ddp_model(dataset['input'])
    labels = dataset['label'].to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    cleanup()


def main(args):
    # load sample dataset
    dataset = {'input': torch.randn(20,10),
               'label': torch.randn(20,5)}
    # train
    '''
    spawn nprocs processes that run fn with args
    process index passed to fn
    ex) below function spawn demo_fn(i, world_size) for i in range(world_size)
    '''
    world_size = args.world_size if args.world_size else torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size,dataset),
             nprocs=world_size,
             join=True)


def get_args():
    parser = argparse.ArgumentParser(description='option for AV training.')
    parser.add_argument('-w','--world_size',
                         default=0, type=int, help='Configurate the number of GPUs')
    parser.add_argument('-d','--data_folder',
                         action = 'store_true', help='Data folder path') 
    args = parser.parse_args()
    return args
  

if __name__ == '__main__':
    args = get_args()
    main(args)