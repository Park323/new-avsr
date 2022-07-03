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


def sentence_to_target(sentence, char2id, unit='character'):
    # tokenize
    if unit=='character':
        _sentence = sentence
    elif unit=='grapheme':
        _sentence = char2grp(sentence)
    target = str()
    
    for ch in _sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            print(f"KeyError Occured, Key : '{ch}', sentence : '{sentence}'")
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
        for video_path, audio_path, transcript in zip(*datasets[usage]):
            char_id_transcript = sentence_to_target(transcript, char2id)
            f1.write(f'{video_path}\t{audio_path}\t{transcript}\t{char_id_transcript}\n')
        f1.close()
        
    
def num2kor(num):
    num = int(num)
    unit = ['일','만','억','조']
    sub_unit = ['일','십','백','천']
    nums = '영일이삼사오육칠팔구'
    string = ''
    
    if num == 0:
        return nums[num]
    if num == 10000:
        return unit[1]
        
    for i in range(len(unit)-1, -1, -1):
        k, num = divmod(num, 10**(4*i))
        if k==0: continue
        for j in range(3, -1, -1):
            l, k = divmod(k, 10**j)
            if l > 0:
                if l > 1 or j == 0:
                    string += nums[l]
                if j > 0: string += sub_unit[j]
        if i > 0:
            string += unit[i]
            string += ' '
    return string
    
    
        
def unzip_groups(f, char2id, unit, video_path, audio_path, transcript):
    #pdb.set_trace()
    pattern = '(\(([^(/)]+)\)([^(/)]*))\/?(\(([^(/)]+)\)(\3)?)'
    if re.search(pattern, transcript):
        for group_num in [1,4]:
            _transcript = re.sub(pattern, f"\{group_num}", transcript)
            _transcript = re.sub('[(/)]', "", _transcript)
            unzip_groups(f, char2id, unit, video_path, audio_path, _transcript)
    else:
        char_id_transcript = sentence_to_target(transcript, char2id, unit)
        f.write(f'{video_path}\t{audio_path}\t{transcript}\t{char_id_transcript}\n')

        
def generate_script_from_path(path, unit='character'):
    print('create_script started..')
    
    with open(path) as f:
        paths = f.readlines()
        
    char2id, id2char = load_label(f"./avsr/vocabulary/kor_{unit}.csv")
    
    save_path = path.replace('.txt',f'_with_{unit}.txt')
    with open(save_path, 'w') as f:
        for _paths in paths:
            units = _paths.strip('\n').split('\t')
            if len(units)!=3:
                print(units) 
                continue
            video_path, audio_path, transcript = units
            
            transcript = re.sub('\xa0',' ',transcript) # \xa0 : space
            transcript = re.sub('[Xx]',' ',transcript) # x : mute
            transcript = re.sub('[%]','퍼센트',transcript) # % : percent
            transcript = re.sub('\d+', lambda x: num2kor(x.group(0)), transcript) # number
            unzip_groups(f, char2id, unit, video_path, audio_path, transcript) # (아기씨)/(애기씨) (안돼써)(안됐어) (그런)게/(그러)게
            
    return save_path


def preprocess(args):
    wav_txt_path = args.wav_txt_folder
    video_path  = args.video_folder
    
    print('preprocess started..')
    
    # call redundanct speakers
    with open('data/redundant_speaker_group.pkl', 'rb') as f:
        redundant = pickle.load(f)
    
    def cleared_len(x):
        length = 0
        for speaker in x:
          condition = f"lip_{args.side}_{args.noise}_{args.sex}_0{args.age}_{speaker}_{args.angle}_{args.index}"
          dataset_path_audio = f'{wav_txt_path}/{condition}/*.wav'
          length += len(glob.glob(dataset_path_audio))
        return length
    
    total_num = 0
    for speaker in redundant:
        total_num += cleared_len(speaker)         
    train_num, test_num = total_num * 0.8, total_num * 0.1
    
    redundant = sorted(redundant, key=lambda x: -cleared_len(x))
    
    # ---------split exclusively--------- #
    datasets = {'Train':([],[],[]),
                'Test' :([],[],[]),
                'Valid':([],[],[]),}
    key = 'Train'
    for speaker_group in redundant:
        for speaker in speaker_group:
            condition = f"lip_{args.side}_{args.noise}_{args.sex}_0{args.age}_{speaker}_{args.angle}_{args.index}"
            dataset_path_audio = f'{wav_txt_path}/{condition}/*.wav'
            [datasets[key][1].append(value) for value in sorted(glob.glob(dataset_path_audio))]
        if len(datasets['Train'][1]) >= train_num:
            key = 'Test'
        if key == 'Test' and len(datasets['Test'][1]) >= test_num:
            key = 'Valid'
    
    for key in ['Train','Test','Valid']:
        for path in datasets[key][1]:
            # choose angle
            if args.angle != "A":
                splited = path.split('_')
                splited[-2] = args.angle
                path = '_'.join(splited)
            path = path.replace(wav_txt_path, video_path)
            path = path[:-4] + '.npy'
            datasets[key][0].append(path)
        
        for file_ in datasets[key][1]:
            txt_file_ = file_.replace('.wav','.txt')
            with open(txt_file_, "r", encoding='utf-8') as f:
                raw_sentence = f.read().strip()
            datasets[key][2].append(raw_sentence)
        
        print(len(datasets[key][1]), end=' ')
    print()
    
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