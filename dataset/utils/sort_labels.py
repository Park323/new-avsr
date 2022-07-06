import random

def _token_length(x):
    return len(x.split('\t')[-1].split())

def check_limit(x, l):
    if _token_length(x) <= l:
        return True
    else:
        return False 

def sort_it(Path:str, Limit:int = None, shuffle = False)->None:
    with open(Path, 'r', encoding='utf8') as f:
        labels = f.readlines()
    
    labels = sorted(labels, key=_token_length)
    if Limit is not None:
        labels = list(filter(lambda x : check_limit(x, Limit), labels))
    if shuffle:
        random.shuffle(labels)
    
    with open(Path.replace('.txt','_sorted.txt'), 'w', encoding='utf8') as f:
        for label in labels:
            f.write(label)