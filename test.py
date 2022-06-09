import os
import json
import tqdm
import pickle

#with open('data/redundant_speaker_group.pkl','rb') as f:
#    redun = pickle.load(f)
#    
#new_redun = []
#total = 0
#for _redun in redun:
#    for x in _redun:
#        new_redun.append(x)
#        total += 1
#print(len(set(new_redun)))
#print(total)
#
#import glob
#wav_txt_path = '/home/data2/nia_wav_txt_1'
#data_path = f"lip_*_*_*_0*_*_A_*"
#all = glob.glob(wav_txt_path+'/'+data_path)
#print(len(all))
#for name in tqdm.tqdm(set(new_redun)):
#    data_path = f"lip_*_*_*_0*_{name}_A_*"
#    partial = glob.glob(wav_txt_path+'/'+data_path)
#    for n in partial:
#        i = 0
#        while i < len(all):
#            if n == all[i]:
#                all.pop(i)
#            else:
#                i += 1
import pdb
#pdb.set_trace()
#print(all)


#########  find speakers  #################
#
#
#def show_contents(mode, n, dataset):
#    r = re.compile(f'.*_._{n}_._.._...._._...[.]mp4')
#    noiseset = list(filter(r.match, dataset))
#    for exp in ['C','E']:
#        r2 = re.compile(f'.*_._._._.._{exp}..._._...[.]mp4')
#        expset = list(filter(r2.match, noiseset))
#        for gender in ['F','M']:
#            r3 = re.compile(f'.*_._._{gender}_.._...._._...[.]mp4')
#            genderset = list(filter(r3.match, expset))
#            print(f"{mode} {n} {exp} {gender}: {len(genderset)}")
#
#def show_speaker(mode, dataset):
#    for n in range(1,7):
#        r = re.compile(f'.*_._{n}_._.._...._._...[.]mp4')
#        noiseset = list(filter(r.match, dataset))
#        for exp in ['C','E']:
#            spk = {}
#            r2 = re.compile(f'.*_._._._.._{exp}..._._...[.]mp4')
#            expset = list(filter(r2.match, noiseset))
#            for path in expset:
#                speaker = path.split('/')[-1].split('_')[-3]
#                spk[speaker] = spk.get(speaker, 0) + 1
#            print(f"{mode} {exp}: {len(spk)}")
#            print(spk.keys())
#
#
#import re
## Dataset exploration
#for mode in ['Train','Valid','Test']:
#    with open(f'data/real_total_{mode}.txt') as f:
#        dataset = f.readlines()
#    #[show_contents(mode, i, dataset) for i in range(1,7)]
#    show_speaker(mode, dataset)
#    

import os
import json
import tqdm

##########  find # of unique sentences  ##########
#
#base_dir = '/home/nas4/NIA/json_bbox_included'
#
#sentences = {'Train':[], 'Test':[],'Valid':[]}
#for mode in ['Valid','Test']:
#    with open(f'data/real_total_{mode}.txt') as f0:
#        dataset = f0.readlines()
#    for fp in tqdm.tqdm(dataset):
#        fp = fp.split("/")[-1][:-5]+'.json'
#        with open(f'{base_dir}/{fp}', encoding='utf-8') as f:
#            labels = json.load(f)
#        for i, label in enumerate(labels[0]['Sentence_info']):
#            sentences[mode].append(label['sentence_text'])
#
#
#for mode in ['Valid','Test','Train']:
#    print(f'{mode} sentences : {len(sentences[mode])}')
#    sentences[mode] = set(sentences[mode])
#
#sentences['Total'] = set() 
#for mode in ['Train','Valid','Test']:
#    print(f'{mode} sentences : {len(sentences[mode])}')
#    for sentence in sentences[mode]:
#        sentences['Total'].add(sentence)
#        
#print(f"Total sentences : {len(sentences['Total'])}")
#

def get_file_size():
    maximums = []
    maximum_paths = []
    limit = 4
    import os
    with open('data/clean_A_Train.txt') as f:
        files = f.readlines()
    with open('data/clean_A_Train_size.txt', 'w') as f:
        for line in files:
            video_path, audio_path, transcripts, kor_transcripts =line.split('\t')
            size = os.path.getsize(video_path)
            if len(maximums)==0 or size > min(maximums):
                if len(maximums) >= limit:
                    pop_idx=maximums.index(min(maximums))
                    maximums.pop(pop_idx)
                    maximum_paths.pop(pop_idx)
                maximums.append(size)
                maximum_paths.append(line)
            f.write(f"{video_path} : {size}\n")
    with open('data/clean_A_Train_maxs.txt','w') as f:
        for size in maximums:
            print(size)
        for line in maximum_paths:
            f.write(line)
get_file_size()

#import glob
#import shutil
#for path in glob.glob('data/pm_*_*'):
#    shutil.move(path, 'data/garbage')