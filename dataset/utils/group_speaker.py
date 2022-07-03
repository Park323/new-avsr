import os
import json
import tqdm
import pickle

speaker_group = []
used = set()

with open('data/checked_redundant_log_2.json', 'r', encoding='utf-8') as f:
    logs = json.load(f)

for sen, group in tqdm.tqdm(logs.items()):
    speakers = [speaker[0].split('_')[-3] for speaker in group]
    #print(sen, len(speakers)) 
    idx = -1
    for speaker in speakers:
        if speaker in used:
            for i, group in enumerate(speaker_group):
                if speaker in group:
                    idx = i
    if idx == -1:
        speaker_group.append(set())
    for speaker in speakers:
        if not speaker in used:
            speaker_group[idx].add(speaker)
            used.add(speaker)
                    
with open('data/redundant_speaker_group_2.pkl', 'wb') as f:
    pickle.dump(speaker_group, f)