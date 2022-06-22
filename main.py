import os
import pdb
import time
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from avsr.model_builder import build_model
from avsr.vocabulary.vocabulary import KsponSpeechVocabulary
from avsr.utils import get_criterion, get_optimizer, get_metric
from dataset.dataset import load_dataset, prepare_dataset, AVcollator
from dataset.sampler import DistributedCurriculumSampler


def setup(rank, world_size, port):
    """
    world_size : number of processes
    rank : this should be a number between 0 and world_size-1
    """
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    '''
    Initialize the process group
    Rule of thumb choosing backend
    NCCL -> GPU training / Gloo -> CPU training
    check table here : https://pytorch.org/docs/stable/distributed.html
    '''
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    
def cleanup():
    dist.destroy_process_group()


def save_checkpoint(model, checkpoint_path, epoch):
    torch.save(model.state_dict(), f"{checkpoint_path}/{epoch:05d}.pt")
    
    
def load_ddp_checkpoint(rank, model, checkpoint_path, epoch):
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(f"{checkpoint_path}/{epoch:05d}.pt", map_location=map_location)
    model.load_state_dict(state_dict)


def show_description(epoch, total_epoch, it, total_it, lr, loss, mean_loss, _time):
    train_time = int(time.time() - _time)
    _sec = train_time % 60
    train_time //= 60
    _min = train_time % 60
    train_time //= 60
    _hour = train_time % 24
    _day = train_time // 24
    desc = f"LOSS {loss:.4f} :: MEAN LOSS {mean_loss:.2f} :: LEARNING_RATE {lr:.8f} :: BATCH [{it}/{total_it}] :: EPOCH [{epoch}/{total_epoch}] :: [{_day:2d}d {_hour:2d}h {_min:2d}m {_sec:2d}s]"
    print(desc, end="\r")


def train(rank, world_size, config, vocab, dataset, port, test=False):
    setup(rank, world_size, port)
    
    save_last = test
    scheduler = 'none' if test else 'noam'
    
    # define a loader
    sampler = DistributedCurriculumSampler(dataset, num_replicas=world_size, rank=rank, drop_last=False)
    
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                            collate_fn = AVcollator(
                                max_len = config['max_len'],
                                use_video = config['use_video'],
                                raw_video = config['raw_video'],), 
                            shuffle = False,
                            sampler = sampler,
                            num_workers=config['num_workers'])
    if rank==0:
        print(f'# of batch for each rank : {len(dataloader)}')
    
    # define a model
    model = build_model(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        architecture=config['architecture'],
        loss_fn=config['loss_fn'],
        front_dim=config['front_dim'],
        encoder_n_layer=config['encoder_n_layer'],
        encoder_d_model=config['encoder_d_model'],
        encoder_n_head=config['encoder_n_head'], 
        encoder_ff_dim=config['encoder_ff_dim'], 
        encoder_dropout_p=config['encoder_dropout_p'],
        decoder_n_layer=config['decoder_n_layer'],
        decoder_d_model=config['decoder_d_model'],
        decoder_n_head=config['decoder_n_head'], 
        decoder_ff_dim=config['decoder_ff_dim'], 
        decoder_dropout_p=config['decoder_dropout_p'],
        rank=rank,
    )
    # move the model to GPU with id rank
    model.to(rank)
    # load state dict
    if config['resume_epoch'] != -1:
        load_ddp_checkpoint(rank, model, checkpoint_path=config['save_dir'], epoch=config['resume_epoch'])
        
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # define a criterion
    criterion = get_criterion(loss_fn=config['loss_fn'], ignore_index=vocab.pad_id, blank_id=vocab.unk_id)
    metric = get_metric(vocab, config['log_path'])
    
    steps_per_epoch = len(dataloader)
    optimizer, scheduler = get_optimizer(
                             ddp_model.parameters(), 
                             learning_rate = config['learning_rate'],
                             epochs = config['epochs'],
                             warmup = float(25000/steps_per_epoch),
                             steps_per_epoch = steps_per_epoch,
                             scheduler = scheduler,
                           )

    for epoch in range(config['resume_epoch']+1, config['epochs']):
        dist.barrier()
        
        sampler.set_epoch(epoch)
        
        ddp_model.train()
        train_start = time.time()
        epoch_total_loss = 0
        for it, (vids, seqs, targets, vid_lengths, seq_lengths, target_lengths) in enumerate(dataloader):
            
            vids = vids.to(rank)
            seqs = seqs.to(rank)
            targets = targets.to(rank)
            target_lengths = target_lengths.to(rank)
            
            """
            Check input sizes
            print()
            print(vids.size())
            print(seqs.size())
            continue
            """
                     
            optimizer.zero_grad()
            
            outputs = ddp_model(vids, vid_lengths,
                                seqs, seq_lengths,
                                targets[:,:-1], target_lengths) # drop eos_id
            
            loss = criterion(outputs=outputs, targets=targets[:,1:], target_lengths=target_lengths) # drop sos_id
            loss.backward()
            optimizer.step()
            scheduler.step(epoch*len(dataloader) + it)
            
            epoch_total_loss += loss.item()
            
            if rank == 0:
                # show description
                show_description(epoch=epoch, 
                                 total_epoch=config['epochs'], 
                                 it = it, 
                                 total_it = len(dataloader), 
                                 lr = scheduler.get_lr()[0],
                                 loss = loss.item(), 
                                 mean_loss = epoch_total_loss/(it+1), 
                                 _time = train_start)
            
        if rank==0 and (not save_last or (epoch+1 == config['epochs'])):
            if not os.path.exists(config['save_dir']):
                os.makedirs(config['save_dir'])
            save_checkpoint(model, config['save_dir'], epoch)
        print()
        print()
        
    cleanup()


def main(args):
    
    # Check Devices
    print("cuda : ", torch.cuda.is_available())
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    vocab = KsponSpeechVocabulary(unit = config['tokenize_unit'])
    
    # load dataset
    dataset = prepare_dataset(
        transcripts_path = config['transcripts_path_train'],
        vocab = vocab,
        use_video = config['use_video'],
        raw_video = config['raw_video'],
        audio_transform_method = config['audio_transform_method'],
        audio_sample_rate = config['audio_sample_rate'],
        audio_normalize = config['audio_normalize'],
        spec_augment = config['spec_augment'],
        freq_mask_para = config['freq_mask_para'],
        freq_mask_num = config['freq_mask_num'],
        time_mask_num = config['time_mask_num'],
        noise_augment = config['noise_augment'],
        noise_path = config['noise_path'],
    )
    print(f"# of data : {len(dataset)}")
    
    # train
    '''
    spawn nprocs processes that run fn with args
    process index passed to fn
    ex) below function spawn demo_fn(i, world_size) for i in range(world_size)
    '''
    world_size = args.world_size if args.world_size else torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size, config, vocab, dataset, args.port, args.test),
             nprocs=world_size,
             join=True)


def get_args():
    parser = argparse.ArgumentParser(description='option for AV training.')
    parser.add_argument('-w','--world_size',
                         default=0, type=int, help='Configurate the number of GPUs')
    parser.add_argument('-c','--config',
                         type=str, help='Configuration Path')
    parser.add_argument('-d','--data_folder',
                         action = 'store_true', help='Data folder path') 
    parser.add_argument('-p','--port',
                         default = '12355', type = str, help='Port number of multi process')
    parser.add_argument('-t','--test', action='store_true',
                         help='Test with small samples')
    args = parser.parse_args()
    return args
  

if __name__ == '__main__':
    args = get_args()
    main(args)
    
