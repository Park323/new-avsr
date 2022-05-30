#-*-coding:utf-8-*-

import re
import csv
import pdb
import torch
import torch.nn.functional as F

# �����ڵ� �ѱ� ���� : 44032, �� : 55203
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
END_CODE = 55203

# �ʼ� ����Ʈ. 00 ~ 18
CHOSUNG_LIST = ['��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��']

# �߼� ����Ʈ. 00 ~ 20
JUNGSUNG_LIST = ['��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��']

# ���� ����Ʈ. 00 ~ 27 + 1(1�� ����)
JONGSUNG_LIST = [None, '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��']


def char2ord(x):
    if len(x)!=1:
        return 0
    else:
        return ord(x)
    

def ord_labeling(df):
    df['ord'] = df['char'].apply(char2ord)
    return df


def char2grp(test_keyword):
    split_keyword_list = list(test_keyword)
    #print(split_keyword_list)

    result = list()
    for keyword in split_keyword_list:
        # �ѱ� ���� check �� �и�
        if re.match('.*[��-�R]+.*', keyword) is not None: # '.*[��-����-�Ӱ�-�R]+.*'
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            result.append(CHOSUNG_LIST[char1])
            #print('�ʼ� : {}'.format(CHOSUNG_LIST[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST[char2])
            #print('�߼� : {}'.format(JUNGSUNG_LIST[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            if char3==0:
                # result.append('<unk>') # att_0
                result.append('<emp>') # att_1, ctc_2
                # att_2
            else:
                result.append(f'#{JONGSUNG_LIST[char3]}')
            #print('���� : {}'.format(JONGSUNG_LIST[char3]))
        else:
            result.append(keyword)
    # result
    return result


def grp2char(JASOlist):
    #id2KR = {code-BASE_CODE+5:chr(code)
    #         for code in range(BASE_CODE, END_CODE + 1)}
    id2KR = {code-BASE_CODE+6:chr(code)
             for code in range(BASE_CODE, END_CODE + 1)} # att_1, ctc_2
    id2KR[0] = '<pad>'
    id2KR[1] = '<sos>'
    id2KR[2] = '<eos>' 
    id2KR[3] = '<unk>'
    id2KR[4] = ' '
    id2KR[5] = '<emp>' # att_1, ctc_2
    # 6 ~ ... => �� ~ .. �R
    KR2id = {key:value for value, key in id2KR.items()}
    
    def reset_count():
        # return 0, 5 # att_0
        return 0, 6 # att_1, ctc_2
    
    result = list()
    chr_count, chr_id = reset_count()

    lists = [CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST]
    nums = [CHOSUNG, JUNGSUNG, 1]
       
    for i, JS in enumerate(JASOlist):
        if chr_count == 2 and len(JS)!=2:
            result.append(id2KR[chr_id])
            chr_count, chr_id = reset_count()
            
        if re.match('.*[��-����-�Ӱ�-�R]+.*', JS) is None:
            result.append(JS)
            continue
                        
        JS = JS.replace('#', '')
        
        if JS in lists[chr_count]:
            chr_id += lists[chr_count].index(JS) * nums[chr_count]
            chr_count += 1
        else:
            chr_count, chr_id = reset_count()
            continue
                
        if chr_count == 3 or (chr_count == 2 and i==len(JASOlist)-1):
            result.append(id2KR[chr_id])
            chr_count, chr_id = reset_count()
    
    result = ''.join(result)
    return result


class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.unk_id = None

    def label_to_string(self, labels):
        raise NotImplementedError


class KsponSpeechVocabulary(Vocabulary):
    def __init__(self, vocab_path, encoding='utf-8'):
        super(KsponSpeechVocabulary, self).__init__()
        
        self.vocab_dict, self.id_dict = self.load_vocab(vocab_path, encoding=encoding)
        self.sos_id = int(self.vocab_dict['<sos>'])
        self.eos_id = int(self.vocab_dict['<eos>'])
        self.pad_id = int(self.vocab_dict['<pad>'])
        self.unk_id = int(self.vocab_dict['<unk>'])
        self.labels = self.vocab_dict.keys()

        self.vocab_path = vocab_path
      

    def __len__(self):
        return len(self.vocab_dict)

    def label_to_string(self, labels, tolist=False):
        """
        Converts label to string (number => Hangeul)
        Args:
            labels (numpy.ndarray): number label
        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """

        if len(labels.shape) == 1:
            sentence = list() if tolist else str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                #elif label.item() == self.unk_id:
                elif label.item() == self.unk_id or label.item()==int(self.vocab_dict.get('<emp>',-1)): # att_1, ctc_2
                    #print(label.item()) # att_1, ctc_2
                    continue
                if tolist:
                    sentence.append(self.id_dict[label.item()])
                else:
                    sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = list() if tolist else str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                #elif label.item() == self.unk_id:
                elif label.item() == self.unk_id or label.item()==int(self.vocab_dict['<emp>']): # att_1, ctc_2
                    #print(label.item()) # att_1, ctc_2
                    continue                    
                if tolist:
                    sentence.append(self.id_dict[label.item()])
                else:
                    sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def load_vocab(self, label_path, encoding='utf-8'):
        """
        Provides char2id, id2char
        Args:
            label_path (str): csv file with character labels
            encoding (str): encoding method
        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
        """
        unit2id = dict()
        id2unit = dict()
        
        try:
            with open(label_path, 'r', encoding=encoding) as f:
                labels = csv.reader(f, delimiter=',')
                next(labels)
                
                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]
                
                #unit2id['<blank>'] = len(unit2id)
                #id2unit[len(unit2id)] = '<blank>'

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))