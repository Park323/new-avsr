import os
import pdb
import yaml
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from avsr.model_builder import build_model
from avsr.vocabulary.vocabulary import KsponSpeechVocabulary
from avsr.utils import get_criterion, get_optimizer, get_metric
from dataset.dataset import load_dataset, prepare_dataset, AVcollator


def show_description(it, total_it, loss, mean_loss, cer, mean_cer, _time):
    train_time = int(time.time() - _time)
    _sec = train_time % 60
    train_time //= 60
    _min = train_time % 60
    train_time //= 60
    _hour = train_time % 24
    _day = train_time // 24
    desc = f"LOSS {loss:.4f} :: MEAN LOSS {mean_loss:.2f} :: CER {cer:.4f} :: MEAN CER {mean_cer:.2f} :: BATCH [{it}/{total_it}] :: [{_day:2d}d {_hour:2d}h {_min:2d}m {_sec:2d}s]"
    print(desc, end="\r")


def load_checkpoint(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(f"{checkpoint_path}", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    

def infer(config, vocab, dataset, device='cpu'):
        
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                            collate_fn = AVcollator(
                                max_len = config['max_len'],
                                use_video = config['use_video'],
                                raw_video = config['raw_video'],), 
                            shuffle = False,
                            num_workers=config['num_workers'])
    
    # define a model
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
    )
    
    # load state dict
    load_checkpoint(model, checkpoint_path=config['model_path'], device=device)
    # move the model to GPU
    model.to(device)
    
    # define a criterion
    criterion = get_criterion(ignore_index=vocab.pad_id, blank_id=vocab.unk_id)
    metric = get_metric(vocab, config['log_path'])
    
    model.eval()
    eval_start = time.time()
    total_loss = 0
    total_cer = 0
    for it, (vids, seqs, targets, vid_lengths, seq_lengths, target_lengths) in enumerate(dataloader):
        vids = vids.to(device)
        seqs = seqs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        outputs = model(vids, vid_lengths,
                            seqs, seq_lengths,
                            targets, target_lengths)
        
        loss = criterion(outputs=outputs, targets=targets, target_lengths=target_lengths)
        output_lengths = torch.tensor([outputs[0].size(1) for _ in range(outputs[0].size(0))]).to(device)
        cer  = metric(outputs=outputs[0], targets=targets, output_lengths=output_lengths, target_lengths=target_lengths)
        
        total_loss += loss.item()
        total_cer  += cer
        
        # show description
        show_description(it = it,
                         total_it = len(dataloader), 
                         loss = loss.item(),
                         mean_loss = total_loss/(it+1),
                         cer = cer,
                         mean_cer = total_cer/(it+1),
                         _time = eval_start)
    print()


def main(args):
    
    # Check Devices
    print("cuda : ", torch.cuda.is_available())
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    vocab = KsponSpeechVocabulary()
    
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])
    
    # load dataset
    dataset = prepare_dataset(
        transcripts_path = config['transcripts_path_test'],
        vocab = vocab,
        use_video = config['use_video'],
        raw_video = config['raw_video'],
        audio_transform_method = config['audio_transform_method'],
        audio_sample_rate = config['audio_sample_rate'],
        audio_normalize = config['audio_normalize'],
    )
    print(f"# of data : {len(dataset)}")
    
    infer(config, vocab, dataset, device=DEVICE)
    
#    DDP inference
#    '''
#    spawn nprocs processes that run fn with args
#    process index passed to fn
#    ex) below function spawn demo_fn(i, world_size) for i in range(world_size)
#    '''
#    world_size = args.world_size if args.world_size else torch.cuda.device_count()
#    mp.spawn(train,
#             args=(world_size, config, vocab, dataset, args.port),
#             nprocs=world_size,
#             join=True)


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
