import os
import json
import tqdm

base_dir = '/home/nas4/NIA/json_bbox_included'
files = os.listdir(base_dir)

sentence_dict = {}

with open('redundant_sentences.txt', 'w') as f0:
    for fp in tqdm.tqdm(files):
        with open(f'{base_dir}/{fp}', encoding='utf-8') as f:
            labels = json.load(f)
        assert len(labels)==1, f"{fp} has {len(labels)} labels"
        speakerID = labels[0]['speaker_info']['speaker_ID']
        for i, label in enumerate(labels[0]['Sentence_info']):
            if sentence_dict.get(label['sentence_text']):
                old = sentence_dict.get(label['sentence_text'])
                if old[0] != speakerID:
                    print(f"{old[0]} and {speakerID} are conflicted with sentence : {label['sentence_text']}")
                    f0.write(f"{old[1]}_{old[2]}, {labels[0]['Video_info']['video_Name']}_{i+1}, {label['sentence_text']}\n")
            else:
                sentence_dict[label['sentence_text']] = (speakerID, labels[0]['Video_info']['video_Name'], i+1)