import os
import pdb
import yaml
import warnings
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from avsr.model_builder import build_model
from avsr.vocabulary.vocabulary import KsponSpeechVocabulary
from avsr.utils import get_criterion, get_optimizer, get_metric
from dataset.dataset import load_dataset, prepare_dataset, AVcollator


warnings.filterwarnings(action='ignore')


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


def train(rank, world_size, dataloader, config, vocab):

    setup(rank, world_size)
    
    # create model and move it to GPU with id rank
    model = build_model(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
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
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    loss_fn = get_criterion(ignore_index=vocab.pad_id)
    metric = get_metric(vocab, config['log_path'])
    optimizer, scheduler = get_optimizer(
                             ddp_model.parameters(), 
                             learning_rate = config['learning_rate'],
                             epochs = config['epochs'],
                             steps_per_epoch = len(dataloader),
                           )
    
    for epoch in tqdm(range(config['epochs'])):
        
        dist.barrier()
        
        model.train()
        progress_bar = tqdm(enumerate(dataloader),ncols=110)
        for i, (vids, seqs, targets, vid_lengths, seq_lengths, target_lengths) in progress_bar:
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
        
        print(loss.item())
        
    cleanup()


def main(args):
    
    # Check Devices
    print("cuda : ", torch.cuda.is_available())
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    vocab = KsponSpeechVocabulary()
    
    # load dataset
    dataset = prepare_dataset(
        transcripts_path = config['transcripts_path_train'],
        vocab = vocab,
        use_video = config['use_video'],
        raw_video = config['raw_video'],
        audio_transform_method = config['audio_transform_method'],
#        audio_sample_rate = config['audio_sample_rate'],
#        audio_n_mels = config['audio_n_mels'],
#        audio_frame_length = config['audio_frame_length'],
#        audio_frame_shift = config['audio_frame_shift'],
        audio_normalize = config['audio_normalize'],
        spec_augment = config['spec_augment'],
        freq_mask_para = config['freq_mask_para'],
        freq_mask_num = config['freq_mask_num'],
        time_mask_num = config['time_mask_num'],
#        noise_augment = config['noise_augment'],
#        noise_path = config['noise_path'],
    )
    
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                            collate_fn = AVcollator(
                                max_len = config['max_len'],
                                use_video = config['use_video'],
                                raw_video = config['raw_video'],), 
                            shuffle=True,
                            num_workers=config['num_workers'])
    print(f'trainset : {len(dataset)}, {len(dataloader)} batches')
    
    # train
    '''
    spawn nprocs processes that run fn with args
    process index passed to fn
    ex) below function spawn demo_fn(i, world_size) for i in range(world_size)
    '''
    world_size = args.world_size if args.world_size else torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size,dataloader,config, vocab),
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
    args = parser.parse_args()
    return args
  

if __name__ == '__main__':
    args = get_args()
    main(args)
    
