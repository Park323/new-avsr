import os

with open("/home/nas3/DB/NIA/wook_nia/expert_test_12120.txt", 'r') as f:
    txt = f.readlines()
    print(f'testset : {len(txt)}')
    for path in txt:
        if not os.path.exists(path.split('\t')[0]):
            print(f'No file : {path}')
        if not os.path.exists(path.split('\t')[1]):
            print(f'No file : {path}')

with open("/home/nas3/DB/NIA/wook_nia/train_nia_0_15_full_3cycle.txt", 'r') as f:
    txt = f.readlines()
    print(f'trainset : {len(txt)}')
    for path in txt:
        if not os.path.exists(path.split('\t')[0]):
            print(f'No file : {path}')
        if not os.path.exists(path.split('\t')[1]):
            print(f'No file : {path}')