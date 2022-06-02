import os
import json
import tqdm
import pickle

with open('data/redundant_speaker_group.pkl','rb') as f:
    redun = pickle.load(f)
    
new_redun = []
total = 0
for _redun in redun:
    for x in _redun:
        new_redun.append(x)
        total += 1
print(len(set(new_redun)))
print(total)

import glob
wav_txt_path = '/home/data2/nia_wav_txt_1'
data_path = f"lip_*_*_*_0*_*_A_*"
all = glob.glob(wav_txt_path+'/'+data_path)
print(len(all))
for name in tqdm.tqdm(set(new_redun)):
    data_path = f"lip_*_*_*_0*_{name}_A_*"
    partial = glob.glob(wav_txt_path+'/'+data_path)
    for n in partial:
        i = 0
        while i < len(all):
            if n == all[i]:
                all.pop(i)
            else:
                i += 1
import pdb
pdb.set_trace()
print(all)