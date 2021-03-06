import os
import re
import pdb
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
        for path in datasets[usage]:
            f1.write(f'{path}\n')
        f1.close()


def preprocess(args):
    file_path = "/home/nas4/NIA/original"
    
    print('preprocess started..')
    
    # call redundanct speakers
    with open('data/redundant_speaker_group_2.pkl', 'rb') as f:
        redundant = pickle.load(f)
    
    total_paths = []
    redun_lengths = []
    for group in tqdm(redundant):
        length = 0
        for speaker in group:
            condition = f"lip_{args.side}_{args.noise}_{args.sex}_0{args.age}_{speaker}_{args.angle}_{args.index}"
            dataset_path = f'{file_path}/*/*/*/{condition}.mp4'
            redun = glob.glob(dataset_path)
            length += len(redun)
            total_paths = total_paths + redun
        redun_lengths.append(length)
    
    print("========= redundant speaker group =========")
    for i, redun in enumerate(redundant):
        print(f"{i} : {redun_lengths[i]}")
    print()
    
    datasets = {'Train':[],
                'Test' :[],
                'Valid':[],}
    indices = {'Train':[0,1,2,5],
                'Test' :[6],
                'Valid':[3,4,7,8,9,10,11,12],}
    for key in ['Train','Test','Valid']:
        for index in indices[key]:
            for speaker in redundant[index]:
                condition = re.compile(f".*{speaker}.*[.]mp4")
                paths = list(filter(condition.match, total_paths))
                for value in sorted(paths):
                    datasets[key].append(value)
    """
    sort_index = sorted(range(len(redundant)), key=lambda k: redun_lengths[k])
    redundant = [redundant[sort_index[i]] for i in range(len(redundant))]
    
    total_num = len(total_paths)
    
    #train_num, test_num = [int(total_num * factor) for factor in [0.91, 0.06]]
    #val_num = total_num - train_num - test_num # 0.03
    
    print(f'total # of chosen dataset : {total_num}')
    
    # ---------split exclusively--------- #
    datasets = {'Train':[],
                'Test' :[],
                'Valid':[],}
    key = 'Valid'
    for i, speaker_group in tqdm(enumerate(redundant)):
        for speaker in speaker_group:
            condition = re.compile(f".*{speaker}.*[.]mp4")
            paths = list(filter(condition.match, total_paths))
            for value in sorted(paths):
                datasets[key].append(value)
        if key=='Valid' and len(datasets['Valid']) >= val_num:
            key = 'Test'
        if key == 'Test' and len(datasets['Test']) >= test_num:
            key = 'Train'
    """
                
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
    parser.add_argument('-ang', '--angle', default='*', type=str, required=False)
    parser.add_argument('-idx', '--index', default="*", type=str, required=False)
    
    parser.add_argument('-sp', '--save_path', type=str, required=True)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = get_args()
    datasets = preprocess(args)
    generate_character_script(datasets, args.save_path)