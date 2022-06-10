import os
import sys
import tqdm
import numpy as np

from augment import get_sample


"""
Check sizes of the feature from video
original audio length 'a' and video feature length 'v' should satisfy this.
"a/480 - 5 <= v <= a/480 - 4
"""
if __name__ == '__main__':
    label_path = sys.argv[1]
    with open(label_path) as f:
        files = f.readlines()
    with open(label_path.replace('.txt','.temp'), 'w') as f:
        for line in files:
            f.write(line)        
    val   = open(label_path, 'w')
    inval = open(label_path.replace('.txt','_invalid.txt'), 'w')
    unpaired = open(label_path.replace('.txt','_unpair.txt'), 'w')
    for path in tqdm.tqdm(files):
        video_path, audio_path, transcript, kor_transcript = path.split('\t')
        
        signal, _ = get_sample(audio_path,resample=14400)
        signal = signal.numpy().reshape(-1,)
        audio_len = signal.shape[0]
        
        try:
            video = np.load(video_path)
        except:
            unpaired.write(path)
            continue
        video = video.transpose(1,0)
        video_len = video.shape[0]
        
        is_valid = video_len == np.ceil(audio_len/480) - 5 or video_len == np.ceil(audio_len/480) - 4
        
        if is_valid:
            val.write(path)
        else:
            inval.write(path)
#            print(f"pm feat length : {video_len}")
#            print(f"audio length : {audio_len}")
#            print('valid :', is_valid)
            
    inval.close()
    val.close()
    unpaired.close()