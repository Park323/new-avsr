import re 
import os
import pdb
import json
import tqdm

#########  find speakers  #################


def show_contents(mode, n, dataset):
    r = re.compile(f'.*_._{n}_._.._...._._...[.]mp4')
    noiseset = list(filter(r.match, dataset))
    for exp in ['C','E']:
        r2 = re.compile(f'.*_._._._.._{exp}..._._...[.]mp4')
        expset = list(filter(r2.match, noiseset))
        for gender in ['F','M']:
            r3 = re.compile(f'.*_._._{gender}_.._...._._...[.]mp4')
            genderset = list(filter(r3.match, expset))
            print(f"{mode} {n} {exp} {gender}: {len(genderset)}")

def show_speaker(mode, dataset):
    for n in range(1,7):
        r = re.compile(f'.*_._{n}_._.._...._._...[.]mp4')
        noiseset = list(filter(r.match, dataset))
        for exp in ['C','E']:
            spk = {}
            r2 = re.compile(f'.*_._._._.._{exp}..._._...[.]mp4')
            expset = list(filter(r2.match, noiseset))
            for path in expset:
                speaker = path.split('/')[-1].split('_')[-3]
                spk[speaker] = spk.get(speaker, 0) + 1
            print(f"{mode} {exp}: {len(spk)}")
            print(spk.keys())

def show_categories_split(file_name):
    for mode in ['Train','Valid','Test']:
        with open(f'data/{file_name}_{mode}.txt') as f:
            dataset = f.readlines()
        [show_contents(mode, i, dataset) for i in range(1,7)]
        #show_speaker(mode, dataset)

##########  find # of unique sentences  ##########
def show_num_unique_sentences(file_name):
    base_dir = '/home/nas4/NIA/json_bbox_included'
    
    sentences = {'Train':[], 'Test':[],'Valid':[]}
    for mode in ['Valid','Test','Train']:
        with open(f'data/{file_name}_{mode}.txt') as f0:
            dataset = f0.readlines()
        for fp in tqdm.tqdm(dataset):
            fp = fp.split("/")[-1][:-5]+'.json'
            with open(f'{base_dir}/{fp}', encoding='utf-8') as f:
                labels = json.load(f)
            for i, label in enumerate(labels[0]['Sentence_info']):
                sentences[mode].append(label['sentence_text'])
    
    
    for mode in ['Valid','Test','Train']:
        print(f'{mode} sentences : {len(sentences[mode])}')
        sentences[mode] = set(sentences[mode])
    
    sentences['Total'] = set() 
    for mode in ['Train','Valid','Test']:
        print(f'{mode} sentences : {len(sentences[mode])}')
        for sentence in sentences[mode]:
            sentences['Total'].add(sentence)
            
    print(f"Total sentences : {len(sentences['Total'])}")


def get_file_size(path, save_path, limit = 4, sep = '\t'):
    import os
    
    maximums = {}
    
    with open(f'{path}') as f:
        files = f.readlines()
    
    for line in files:
        content = line.split(sep)
        if len(content)==4:
            video_path, audio_path, transcripts, kor_transcripts = content 
        else:
            video_path, audio_path, transcripts = content
        size = os.path.getsize(video_path) + os.path.getsize(audio_path)
        minimum = min(maximums.values()) if len(maximums)!=0 else 0
        if size > minimum:
            if len(maximums) >= limit:
                maximums.pop(list(maximums.keys())[-1])
            maximums[tuple(content)] = size
            maximums = dict(sorted(maximums.items(), key= lambda it : it[1], reverse=True))
    
    with open(f'{save_path}','w') as f:
        for content, size in maximums.items():
            print(size)
            f.write('\t'.join(content))
            
if __name__=='__main__':
    show_categories_split("1_A_DLR_with_character")
    show_num_unique_sentences("1_A_DLR_with_character")
    #get_file_size()