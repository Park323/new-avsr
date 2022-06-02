import os
import re
import glob
import random
import pickle
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

import pdb

from avsr.vocabulary.vocabulary import char2grp


def load_label(filepath, encoding='utf-8'):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding=encoding)
    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    #freq_list = ch_labels.get("freq", id_list)
    #ord_list = ch_labels["ord"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()
    
    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            print(f"KeyError Occured, Key : '{ch}'")
            continue

    return target[:-1]


def target_to_sentence(target, id2char):
    sentence = ""
    targets = target.split()

    for n in targets:
        sentence += id2char[int(n)]
    return sentence


def generate_character_script(datasets, save_path, valid_rate=0.2):
    print('create_script started..')
    
    char2id, id2char = load_label("./avsr/vocabulary/kor_characters.csv")
    
    for usage in datasets.keys():
        _save_path = save_path.replace('.txt',f'_{usage}.txt')
        f1 = open(_save_path, "w")
        for video_path, audio_path, transcript in zip(datasets[usage]):
            char_id_transcript = sentence_to_target(transcript, char2id)
            f1.write(f'{video_path}\t{audio_path}\t{transcript}\t{char_id_transcript}\n')
        f1.close()


def preprocess(args):
    wav_txt_path = args.wav_txt_folder
    video_path  = args.video_folder
    
    print('preprocess started..')
    transcripts=[]
    
    # define data paths
    data_path = f"lip_{args.side}_{args.noise}_{args.sex}_0{args.age}_{args.expert}{args.speaker_id}_A_{args.index}"
    folders = glob.glob(wav_txt_path+'/'+data_path)
    print(f"{len(folders)} folders detected...")
    
    # count the number of files
    dataset_path_audio = f'{wav_txt_path}/{data_path}/*.wav'
    total_num =  glob.glob(dataset_path_audio)
    train_num, test_num = total_num * 0.8, total_num * 0.1
    
    # call redundanct speakers
    with open('data/redundant_speaker_group.pickle') as f:
        redundant = pickle.load(f)
    
    redundant = sorted(redundant, key=lambda x: -len(x))
    for speaker_group in redundant:
        for speaker in speaker_group:
            used_speakers.add(speaker)

            dataset_path_audio = wav_txt_path + f'/*_{speaker}_*/*.wav'
            audio_paths = audio_paths + sorted(glob.glob(dataset_path_audio))
            
            for path in audio_paths:
                # choose angle
                if args.angle != "A":
                    splited = path.split('_')
                    splited[-2] = args.angle
                    path = '_'.join(splited)
                
                path = path.replace(wav_txt_path, video_path)
                path = path[:-4] + '.npy'
                video_paths.append(path)
            
            for file_ in tqdm(audio_paths):
                txt_file_ = file_.replace('.wav','.txt')
                with open(txt_file_, "r", encoding='utf-8') as f:
                    raw_sentence = f.read().strip()
                transcripts.append(raw_sentence)
            
            if len(tr_video_paths) >= train_num:
                break
    
    if args.split:
        datasets = {
            "Train":(tr_video_paths, tr_audio_paths, tr_transcripts),
            "Valid":(vl_video_paths, vl_audio_paths, vl_transcripts),
            "Test" :(tt_video_paths, tt_audio_paths, tt_transcripts),
        }
    else:
        datasets = {
            "Train":(tr_video_paths, tr_audio_paths, tr_transcripts),
        }
    
    return datasets


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split', default=True, type=bool)
    
    parser.add_argument('-s', '--side', default="*", type=str, required=False)
    parser.add_argument('-n', '--noise', default="*", type=str, required=False)
    parser.add_argument('-sx', '--sex', default="*", type=str, required=False)
    parser.add_argument('-age', '--age', default="*", type=str, required=False)
    parser.add_argument('-ex', '--expert', default="*", type=str, required=False)
    parser.add_argument('-id', '--speaker_id', default="*", type=str, required=False)
    parser.add_argument('-ang', '--angle', default='A', type=str, required=False)
    parser.add_argument('-idx', '--index', default="*", type=str, required=False)
    
    parser.add_argument('-d', '--wav_txt_folder', type=str, required=True)
    parser.add_argument('-v', '--video_folder', type=str, required=True)
    parser.add_argument('-sp', '--save_path', type=str, required=True)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = get_args()
    datasets = preprocess(args)
    generate_character_script(datasets, args.save_path)