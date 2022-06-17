import re 
import os
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


def get_file_size(path, save_path, sep='\t'):
    import os
    
    maximums = []
    maximum_paths = []
    limit = 4
    
    with open(f'{path}') as f:
        files = f.readlines()
    
    for line in files:
        if len(line.split(sep))==4:
            video_path, audio_path, transcripts, kor_transcripts = line.split(sep)
        else:
            video_path, audio_path, transcripts = line.split(sep)
        size = os.path.getsize(video_path)
        if len(maximums)==0 or size > min(maximums):
            if len(maximums) >= limit:
                pop_idx=maximums.index(min(maximums))
                maximums.pop(pop_idx)
                maximum_paths.pop(pop_idx)
            maximums.append(size)
            maximum_paths.append(line)
    
    with open(f'{save_path}','w') as f:
        for size in maximums:
            print(size)
        for line in maximum_paths:
            f.write(line)
            
if __name__=='__main__':
    show_categories_split("1_A_DLR_with_character")
    show_num_unique_sentences("1_A_DLR_with_character")
    #get_file_size()