import os
import pdb
import yaml
import time
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from avsr.model_builder import build_model
from avsr.utils import get_criterion, get_optimizer, get_metric, select_search
from avsr.vocabulary.vocabulary import KsponSpeechVocabulary
from dataset.dataset import load_dataset, prepare_dataset, AVcollator
from dataset.sampler import DistributedCurriculumSampler


mp = mp.get_context('spawn')


def show_description(it, total_it, ger, mean_ger, cer, mean_cer, wer, mean_wer, _time):
    train_time = int(time.time() - _time)
    _sec = train_time % 60
    train_time //= 60
    _min = train_time % 60
    train_time //= 60
    _hour = train_time % 24
    _day = train_time // 24
    desc = f"GER {ger:.4f} :: MEAN GER {mean_ger:.4f} :: CER {cer:.4f} :: MEAN CER {mean_cer:.4f} :: WER {wer:.4f} :: MEAN WER {mean_wer:.4f} :: BATCH [{it}/{total_it}] :: [{_day:2d}d {_hour:2d}h {_min:2d}m {_sec:2d}s]"
    print(desc, end="\r")


def load_checkpoint(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(f"{checkpoint_path}", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    

def infer(rank, world_size, config, model, vocab, dataset, scores, device='cpu', port=12345):
    if world_size > 1:
        # define a loader
        sampler = DistributedCurriculumSampler(dataset, num_replicas=world_size, rank=rank, drop_last=False)
    else:
        sampler = None
            
    dataloader = DataLoader(dataset=dataset, batch_size=1,
                            collate_fn = AVcollator(
                                max_len = config['max_len'],
                                use_video = config['use_video'],
                                raw_video = config['raw_video'],
                            ), 
                            shuffle = False,
                            sampler = sampler,
                            num_workers=config['num_workers'])
    
    # define a criterion
    criterion = get_criterion(loss_fn=config['loss_fn'], ignore_index=vocab.pad_id, blank_id=vocab.unk_id)
    metric_ger = get_metric(vocab, config['log_path'], unit=config['tokenize_unit'], error_type='ger')
    metric_cer = get_metric(vocab, config['log_path'], unit=config['tokenize_unit'], error_type='cer')
    metric_wer = get_metric(vocab, config['log_path'], unit=config['tokenize_unit'], error_type='wer')
    search = select_search(config['search_method'])
    
    model.eval()
    eval_start = time.time()
    total_ger = 0
    total_cer = 0
    total_wer = 0
    for it, (vids, seqs, targets, vid_lengths, seq_lengths, target_lengths) in enumerate(dataloader):
        vids = vids.to(device)
        seqs = seqs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        with torch.no_grad():
            outputs = search(
                model, 
                vids, vid_lengths,
                seqs, seq_lengths,
                max_len=config['max_len'],
                vocab_size=len(vocab),
                beam_size = config['beam_size'],
                pad_id=vocab.pad_id,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                blank_id=vocab.unk_id,
                ctc_rate=config['ctc_rate'],
                device=device,
            )
      
        output_lengths = torch.tensor([len(outputs[0])]).to(device)
        ger = metric_ger(outputs=outputs, targets=targets[:,1:], output_lengths=output_lengths, target_lengths=target_lengths) if metric_ger else 0
        cer = metric_cer(outputs=outputs, targets=targets[:,1:], output_lengths=output_lengths, target_lengths=target_lengths)
        wer = metric_wer(outputs=outputs, targets=targets[:,1:], output_lengths=output_lengths, target_lengths=target_lengths)
        
        scores += torch.tensor([1, ger, cer, wer])
        
        if rank==0:
            # show description
            show_description(it = int(scores[0].item()),
                             total_it = len(dataset), 
                             ger = ger,
                             mean_ger = scores[1]/scores[0],
                             cer = cer,
                             mean_cer = scores[2]/scores[0],
                             wer = wer,
                             mean_wer = scores[3]/scores[0],
                             _time = eval_start)


def main(args):
    
    # Check Devices
    print("cuda : ", torch.cuda.is_available())
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    vocab = KsponSpeechVocabulary(unit = config['tokenize_unit'])
    
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
    
    # Assertion
    assert config['num_mp'] <= len(dataset), "num_mp should be equal or smaller than size of dataset!!"
    
    # DDP inference
    '''
    spawn nprocs processes that run fn with args
    process index passed to fn
    ex) below function spawn demo_fn(i, world_size) for i in range(world_size)
    '''
    world_size = config['num_mp']
    processes = []
    
    scores = torch.tensor([0.,0.,0.,0.])
    scores.share_memory_()
    for i in range(world_size):
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
            verbose= i==0,
        )
        # load state dict
        load_checkpoint(model, checkpoint_path=config['model_path'], device=DEVICE)
        # move the model to GPU
        model.to(DEVICE)
        
        if world_size > 1:
            process = mp.Process(target=infer,
                                 args=(i, world_size, config, model, vocab, dataset, scores, DEVICE, args.port),)
            process.start()
            processes.append(process)
        else:
            infer(i, world_size, config, model, vocab, dataset, scores, DEVICE, args.port)
        
    if world_size > 1:
        for process in processes:
            process.join()
    
    print()
    print()
    print(f"[Results]")
    print(f"Grapheme Error Rate  : {100*scores[1]/scores[0]:.2f}%")
    print(f"Character Error Rate : {100*scores[2]/scores[0]:.2f}%")
    print(f"Word Error Rate      : {100*scores[3]/scores[0]:.2f}%")


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
