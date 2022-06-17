def _token_length(x):
    return len(x.split('\t')[-1].split())

def sort_it(Path:str)->None:
    with open(Path, 'r', encoding='utf8') as f:
        labels = f.readlines()
    labels = sorted(labels, key=_token_length)
    with open(Path.replace('.txt','_sorted.txt'), 'w', encoding='utf8') as f:
        for label in labels:
            f.write(label)