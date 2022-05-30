import os
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from avsr.model_builder import build_model
from avsr.utils import get_criterion, get_optimizer, get_metric
from dataset.dataset import load_dataset
        

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


def train(rank, world_size, dataset, config, vocab):

    setup(rank, world_size)
    
    # create model and move it to GPU with id rank
    model = build_model(
        decoder_d_model=config['decoder_d_model'],
        decoder_n_head=config['decoder_n_head'], 
        decoder_ff_dim=config['decoder_ff_dim'], 
        decoder_dropout_p=config['decoder_dropout_p'],
        encoder_d_model=config['encoder_d_model'],
        encoder_n_head=config['encoder_n_head'], 
        encoder_ff_dim=config['encoder_ff_dim'], 
        encoder_dropout_p=config['encoder_dropout_p']    
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    loss_fn = get_criterion(ignore_index=vocab.pad_id)
    metric = get_metric(vocab, config['log_path'])
    optimizer = get_optimizer(
                              ddp_model.parameters(), 
                              learning_rate = config['learning_rate'],
                              epochs = config['epochs'],
                              steps_per_epoch = len(dataloader),
                              )
    
    for epoch in tqdm(range(epochs)):
        pbar = tqdm(enumerate(dataloader))
        for i, (vids, seqs, targets, vid_lengths, seq_lengths, target_lengths) in pbar:
            vids = vids.to(rank)
            seqs = seqs.to(rank)
            targets = targets.to(rank)
            
            optimizer.zero_grad()
            
            outputs = ddp_model(vids, vid_lengths,
                                seqs, seq_lengths,
                                targets, target_lengths)
            
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
    cleanup()


def main(args):
    
    # Check Devices
    print("cuda : ", torch.cuda.is_available())
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # load dataset
    trainset = prepare_dataset(config, config.train.transcripts_path_train, vocab, Train=True)
    collate_fn = lambda batch: _collate_fn(batch, config.max_len, config.use_video, config.raw_video)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.batch_size,
                                               collate_fn = collate_fn, shuffle=True,
                                               num_workers=config.num_workers)
    print(f'trainset : {len(trainset)}, {len(train_loader)} batches')
    
    # train
    
    config = None # dddddd
    vocab = KsponSpeechVocabulary(config.vocab_label)
    
    '''
    spawn nprocs processes that run fn with args
    process index passed to fn
    ex) below function spawn demo_fn(i, world_size) for i in range(world_size)
    '''
    world_size = args.world_size if args.world_size else torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size,dataset,config, vocab),
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
    
