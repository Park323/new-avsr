import os
import re
import glob
import random
import pickle
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np


def generate_character_script(datasets, save_path, valid_rate=0.2):
    for usage in datasets.keys():
        _save_path = save_path.replace('.txt',f'_{usage}.txt')
        f1 = open(_save_path, "w")
        for audio_path in datasets[usage]:
            f1.write(f'{audio_path}\n')
        f1.close()


def preprocess(args):
    file_path = "/home/nas4/NIA/original/1(소음없음)"
    
    print('preprocess started..')
    
    # call redundanct speakers
    with open('data/redundant_speaker_group.pkl', 'rb') as f:
        redundant = pickle.load(f)
    
    total_paths = []
    redun_lengths = []
    for group in tqdm(redundant):
        for speaker in group:
          condition = f"lip_{args.side}_{args.noise}_{args.sex}_0{args.age}_{speaker}_A_{args.index}"
          dataset_path_audio = f'{file_path}/*/*/{condition}.wav'
          redun = glob.glob(dataset_path_audio)
          redun_lengths.append(len(redun))
          total_paths = total_paths + redun
    
    sort_index = sorted(range(len(redundant)), key=lambda k: -redun_lengths[k])
    redundant = [redundant[sort_index[i]] for i in range(len(redundant))]
    
    total_num = len(total_paths)
    train_num, test_num = total_num * 0.8, total_num * 0.1
    print(f'total # of chosen dataset : {total_num}')
    
    datasets = {'Train':[],
                'Test' :[],
                'Valid':[],}
    key = 'Train'
    for speaker_group in tqdm(redundant):
        for speaker in speaker_group:
            condition = re.compile(f".*{speaker}.*[.]wav")
            paths = list(filter(condition.match, total_paths))
            for value in sorted(paths):
                datasets[key].append(value)
                if key == 'Test' and len(datasets['Test']) >= test_num:
                    key = 'Valid'
        if key=='Train' and len(datasets['Train']) >= train_num:
            key = 'Test'
                
    for key in ['Train','Test','Valid']:    
        print(len(datasets[key]), end=' ')
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
    
    parser.add_argument('-sp', '--save_path', type=str, required=True)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = get_args()
    datasets = preprocess(args)
    generate_character_script(datasets, args.save_path)