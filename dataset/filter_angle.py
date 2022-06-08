import re
import os
import pdb

for mode in ['Train', 'Valid', 'Test']:
    with open(f'data/real_total_{mode}.txt') as f:
        paths = f.readlines()
    r = re.compile('.*lip_._1_._.._...._A_.*')
    filtered_paths = list(filter(r.match, paths))
            
    with open(f'data/only_A_{mode}.txt', 'w') as f:
        for path in filtered_paths:
            f.write(path)