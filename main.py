import os
import pdb
import time
import yaml
import random
import argparse
from tqdm import tqdm

import wandb
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True

from avsr.model_builder import build_model
from avsr.vocabulary.vocabulary import KsponSpeechVocabulary
from avsr.utils import get_criterion, get_optimizer
from avsr.log_writer import MyWriter
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
    
    import platform
    backend = 'gloo' if platform.system() == 'Windows' else 'nccl'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    
def cleanup():
    dist.destroy_process_group()


def save_checkpoint(model, checkpoint_path, epoch):
    torch.save(model.state_dict(), f"{checkpoint_path}/{epoch:05d}.pt")
    
    
def load_ddp_checkpoint(rank, model, checkpoint_path, epoch):
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(f"{checkpoint_path}/{epoch:05d}.pt", map_location=map_location)
    model.load_state_dict(state_dict)


def show_description(epoch, total_epoch, it, total_it, lr, loss, mean_loss, _time, ctc_loss=None, att_loss=None, ctc_mean_loss=None, att_mean_loss=None):
    train_time = int(time.time() - _time)
    _sec = train_time % 60
    train_time //= 60
    _min = train_time % 60
    train_time //= 60
    _hour = train_time % 24
    _day = train_time // 24
    if ctc_loss:
        desc = f"LOSS(Total/CTC/ATT) {loss:.4f}/{ctc_loss:.4f}/{att_loss:.4f} :: MEAN LOSS {mean_loss:.4f}/{ctc_mean_loss:.4f}/{att_mean_loss:.4f} :: LEARNING_RATE {lr:.8f} :: BATCH [{it}/{total_it}] :: EPOCH [{epoch}/{total_epoch}] :: [{_day:2d}d {_hour:2d}h {_min:2d}m {_sec:2d}s]"
    else:
        desc = f"LOSS {loss:.4f} :: MEAN LOSS {mean_loss:.4f} :: LEARNING_RATE {lr:.8f} :: BATCH [{it}/{total_it}] :: EPOCH [{epoch}/{total_epoch}] :: [{_day:2d}d {_hour:2d}h {_min:2d}m {_sec:2d}s]"
    print(desc, end="\r")


def train(rank, world_size, config, vocab, dataset, port):
    setup(rank, world_size, port)
    
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
        # weight & biases
        wandb.init(
            # Set the project where this run will be logged
            project="avsr-nia", 
            entity="park323",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_{config['save_dir']}", 
            # Track hyperparameters and run metadata
            config={
            "learning_rate": config['learning_rate'],
            "ctc_rate": config['ctc_rate'],
            "domain": config['architecture'],
            "unit": config['tokenize_unit'],
            "decoder": config['loss_fn'],
            "scheduler" : config['scheduler'],
            "epochs": config['epochs'],
        })
        # tensorboardX
        summary = MyWriter('results/tensorboard/'+config['save_dir'])
    
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
        verbose=rank==0,
    )
    # move the model to GPU with id rank
    model.to(rank)
    # load state dict
    if config['resume_epoch'] != -1:
        load_ddp_checkpoint(rank, model, checkpoint_path='results/'+config['save_dir'], epoch=config['resume_epoch'])
        
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # define a criterion
    criterion = get_criterion(loss_fn=config['loss_fn'], ignore_index=vocab.pad_id, blank_id=vocab.unk_id)
    
    steps_per_epoch = len(dataloader)
    optimizer, scheduler = get_optimizer(
                             ddp_model.parameters(), 
                             learning_rate = config['learning_rate'],
                             init_lr = config['init_lr'],
                             final_lr = config['final_lr'],
                             gamma = config['gamma_lr'],
                             epochs = config['epochs'],
                             warmup = config['warmup']/steps_per_epoch,
                             steps_per_epoch = steps_per_epoch,
                             scheduler = config['scheduler'],
                           )
    
    if rank==0:
        for epoch in range(config['resume_epoch']+1):
            scheduler.step(on='epoch')
    
    for epoch in range(config['resume_epoch']+1, config['epochs']):
        dist.barrier()
        
        ## Very Very Important
        sampler.set_epoch(epoch)
        
        ddp_model.train()
        train_start = time.time()
        epoch_total_loss = 0
        epoch_ctc_loss = 0
        epoch_att_loss = 0
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
            """
            
            optimizer.zero_grad()
            
            outputs = ddp_model(vids, vid_lengths,
                                seqs, seq_lengths,
                                targets[:,:-1], target_lengths) # drop eos_id
                                
            loss = criterion(outputs=outputs, targets=targets[:,1:], target_lengths=target_lengths) # drop sos_id
            if isinstance(loss, tuple):
                loss[0].backward()
                ctc_loss = loss[1]
                att_loss = loss[2]
                loss = loss[0].item()
                epoch_total_loss += loss
                epoch_ctc_loss += ctc_loss
                epoch_att_loss += att_loss
            else:
                loss.backward()
                ctc_loss = None
                att_loss = None
                loss = loss.item()
                epoch_total_loss += loss
            
            if loss > 10:
                print(f'{epoch} epoch {it}th Loss :', loss)
                print('ctc',ctc_loss)
                print('att',att_loss)
                print(targets)

            if config['max_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_norm"])

            optimizer.step()

            if rank == 0:
                cur_lr = scheduler.get_lr()[0]
                scheduler.step(on='step', step = epoch*steps_per_epoch + it)
                # show description
                show_description(
                                epoch=epoch, 
                                total_epoch=config['epochs'], 
                                it = it, 
                                total_it = len(dataloader), 
                                lr = cur_lr,
                                loss = loss, 
                                mean_loss = epoch_total_loss/(it+1), 
                                _time = train_start,
                                ctc_loss = ctc_loss,
                                att_loss = att_loss,
                                ctc_mean_loss = epoch_ctc_loss/(it+1), 
                                att_mean_loss = epoch_att_loss/(it+1), 
                                )
                wandb.log({
                    "train/epoch":epoch,
                    "train/loss":loss,
                    "train/ctc_loss":ctc_loss,
                    "train/att_loss":att_loss,
                    "train/lr":cur_lr,
                })
                # Summary writer
                summary.save_log(epoch*steps_per_epoch+it, loss, ctc_loss, att_loss, cur_lr)
            
        if rank==0:
            scheduler.step(on='epoch', loss = loss)
            if not os.path.exists('results/'+config['save_dir']):
                os.makedirs('results/'+config['save_dir'])
            save_checkpoint(model, 'results/'+config['save_dir'], epoch)

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

    # fix the seed
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])

    vocab = KsponSpeechVocabulary(unit = config['tokenize_unit'])
    
    # load dataset
    dataset = prepare_dataset(
        transcripts_path = config['transcripts_path_train'],
        vocab = vocab,
        use_video = config['use_video'],
        raw_video = config['raw_video'],
        audio_transform_method = config['audio_transform_method'],
        audio_sample_rate = config['audio_sample_rate'],
        audio_n_mels = config['audio_n_mels'],
        audio_frame_length = config['audio_frame_length'],
        audio_frame_shift = config['audio_frame_shift'],
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
            args=(world_size, config, vocab, dataset, args.port),
            nprocs=world_size,
            join=True)

    print()
    print("wandb finish")    
    # Mark the run as finished (useful in Jupyter notebooks)
    wandb.finish()

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
    args = parser.parse_args()
    return args
  

if __name__ == '__main__':
    args = get_args()
    main(args)
    
