def match(file_name, split_header):
    with open(file_name, 'r') as f:
        contents = f.readlines()
    
    split_std = {}
    for mode in ['Train','Test','Valid']:
        split_std[mode]=[]
        with open(split_header+f"_{mode}.txt") as f:
            lines = f.readlines()
            for line in lines:
                split_std[mode].append(line.strip('\n').split('/')[-1].replace('.mp4', ''))
    
    new_fs = {}
    for mode in ['Train','Test','Valid']:
        new_fs[mode] = open(file_name.replace(".txt", f"_{mode}.txt"), 'w')
    
    for line in contents:
        group = line.strip('\n').split('\t')[0].split('/')[-2]
        for mode in ['Train','Test','Valid']:
            if group in split_std[mode]:
                new_fs[mode].write(line)
    