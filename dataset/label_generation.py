import os
import re
import glob
import random
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


def generate_character_script(videos_paths, audios_paths, transcripts, save_path, valid_rate=0.2):
    print('create_script started..')
    
    char2id, id2char = load_label("./avsr/vocabulary/kor_characters.csv")
    
    f1 = open(save_path, "w")
    for video_path, audio_path,transcript in zip(videos_paths, audios_paths, transcripts):
        char_id_transcript = sentence_to_target(transcript, char2id)
        f1.write(f'{video_path}\t{audio_path}\t{transcript}\t{char_id_transcript}\n')
    f1.close()


def preprocess(args):
    wav_txt_path = args.wav_txt_folder
    video_path  = args.video_folder
    
    print('preprocess started..')
    transcripts=[]
    
    pattern = f'{wav_txt_path}/*'
    folders = [folder.split('/')[-1] for folder in glob.glob(pattern)]
    if args.side:
        folders = [folder for folder in folders if folder[4]==args.side]
    if args.noise:
        folders = [folder for folder in folders if folder[6]==args.noise]
    if args.sex:
        folders = [folder for folder in folders if folder[8]==args.sex]
    if args.age:
        folders = [folder for folder in folders if folder[11]==args.age]
    if args.expert:
        folders = [folder for folder in folders if folder[13]==args.expert]
    if args.speaker_id:
        folders = [folder for folder in folders if folder[14:17]==args.speaker_id]
    #if args.angle:
    #    folders = [folder for folder in folders if folder[18]==args.angle]
    if args.index:
        folders = [folder for folder in folders if folder[20:]==args.index]
    folders = sorted(folders)
    print(f"{len(folders)} folders detected...")
    
    audio_paths = []
    for folder in folders:
        dataset_path_audio = wav_txt_path + f'/{folder}/*.wav'
        audio_paths = audio_paths + sorted(glob.glob(dataset_path_audio))
    
    video_paths = []
    for path in audio_paths:
        # choose angle
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
    #else:
    #    transcripts.extend(['-']*len(audios_paths))

    return video_paths, audio_paths, transcripts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--side', type=str, required=False)
    parser.add_argument('-n', '--noise', type=str, required=False)
    parser.add_argument('-sx', '--sex', type=str, required=False)
    parser.add_argument('-age', '--age', type=str, required=False)
    parser.add_argument('-ex', '--expert', type=str, required=False)
    parser.add_argument('-id', '--speaker_id', type=str, required=False)
    parser.add_argument('-ang', '--angle', type=str, default='A', required=False)
    parser.add_argument('-idx', '--index', type=str, required=False)
    
    parser.add_argument('-d', '--wav_txt_folder', type=str, required=True)
    parser.add_argument('-v', '--video_folder', type=str, required=True)
    parser.add_argument('-sp', '--save_path', type=str, required=True)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = get_args()
    videos_paths, audios_paths, transcripts = preprocess(args)
    generate_character_script(videos_paths, audios_paths, transcripts, args.save_path)