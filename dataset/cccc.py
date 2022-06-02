import os
import json
import tqdm

with open('data/redundant_speaker_group.json') as f:
    spk = json.load(f)
    
keys = list(spk.keys())
groups = []
used = set()

for key in keys:
    #import pdb
    #pdb.set_trace()
    if key in used:
        for idx, group in enumerate(groups):
            if key in group:
                break
        for value in spk[key]:
            if value in used:
                continue
            else:
                groups[idx].add(value)
    else:
        groups.append({key})
        for subkey in spk[key]:
            groups[-1].add(subkey)
            used.add(subkey)
    used.add(key)
        
total = 0
for group in groups:
    print(group)
    total += len(group)
    
    
print(total)
print(len(keys))