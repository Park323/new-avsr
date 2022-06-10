import os
import json
import tqdm

base_dir = '/home/nas4/NIA/json_bbox_included'
files = os.listdir(base_dir)

sentence_info = {}

for fp in tqdm.tqdm(files):
    with open(f'{base_dir}/{fp}', encoding='utf-8') as f:
        labels = json.load(f)
    assert len(labels)==1, f"{fp} has {len(labels)} labels"
    if labels[0]['Video_info']['video_Name'].split('_')[6]=='A':
        for i, label in enumerate(labels[0]['Sentence_info']):
            if sentence_info.get(label['sentence_text']):
                sentence_info[label['sentence_text']].append((labels[0]['Video_info']['video_Name'], label['ID']))
            else:
                sentence_info[label['sentence_text']] = [(labels[0]['Video_info']['video_Name'], label['ID'])]

with open('checked_redundant_log.json', 'w') as f:
    json.dump(sentence_info, f, ensure_ascii=False) 