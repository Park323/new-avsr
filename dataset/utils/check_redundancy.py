import os
import pdb
import json
import glob
import tqdm

base_dir = '/home/nas4/NIA/json_bbox_included'
files = glob.glob(f"{base_dir}/*.json")

sentence_info = {}

for fp in tqdm.tqdm(files):
    with open(fp, encoding='utf-8') as f:
        labels = json.load(f)
    assert len(labels)==1, f"{fp} has {len(labels)} labels"
    
    if labels[0]['Video_info']['video_Name'].split('_')[6]=='A':
        for i, label in enumerate(labels[0]['Sentence_info']):
            sentence = label['sentence_text'].replace('.','').replace(' ','').strip()
            if sentence_info.get(sentence):
                sentence_info[sentence].append((labels[0]['Video_info']['video_Name'], label['ID']))
            else:
                sentence_info[sentence] = [(labels[0]['Video_info']['video_Name'], label['ID'])]

with open('data/checked_redundant_log_2.json', 'w') as f:
    json.dump(sentence_info, f, ensure_ascii=False) 