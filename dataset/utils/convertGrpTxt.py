import pdb

from avsr.vocabulary.vocabulary import grp2char

def findGraphemeTokens(text):
    result = list()
    isJS = False
    for char in text:
        if isJS:
            result.append('\\'+char)
            isJS = False
        elif char != '\\':
            result.append(char)
        else:
            isJS = True
    return result

def convertText(path, opt='grp2char'):
    with open(path, 'r') as f:
        texts = f.readlines()
    results = []
    for i in range(0,len(texts),3):
        results.append(texts[i:i+3])
    results = sorted(results, key=lambda x: x[0])
    
    with open(path.replace('.txt','_c.txt'), 'w', encoding='utf-8') as f:
        for text in texts:
            tokens = findGraphemeTokens(text)
            f.write(grp2char(tokens))