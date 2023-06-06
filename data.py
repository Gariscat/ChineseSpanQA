import torch
from torch.utils.data import Dataset, DataLoader
import json
from pprint import pprint
from typing import Any, List
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def locate_list_in_list(sup: List, sub: List):
    st, ed = None, None
    for i in range(len(sup)-len(sub)+1):
        ok = True
        for j in range(len(sub)):
            if sup[i+j] != sub[j]:
                ok = False
                break
        if ok:
            st, ed = i, i + len(sub) - 1
    return st, ed


class CMRCDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=1024):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)['data']

        self._data = []
            
        for _ in tqdm(raw_data):
            assert len(_['paragraphs']) == 1  # redundant list
            para_dict = _['paragraphs'][0]
            context = para_dict['context']
            
            for q_dict in para_dict['qas']:
                assert len(q_dict['answers']) == 1  # redundant list
                question = q_dict['question']
                answer_text = q_dict['answers'][0]['text']
                answer_st = q_dict['answers'][0]['answer_start']
                if context[answer_st] != answer_text[0]:  # erroneous annotation
                    continue
        
                input_ids = tokenizer.encode(
                    text=context,
                    text_pair=question,
                    padding='max_length',
                    max_length=max_length,
                )
                answer_ids = tokenizer.encode(
                    text=answer_text,
                    add_special_tokens=False,
                    padding=False,
                )
                assert len(input_ids) == max_length

                st, ed = locate_list_in_list(input_ids, answer_ids)
                if st is None or ed is None:
                    continue

                self._data.append((input_ids, st, ed))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index) -> Any:
        input_ids, st, ed = self._data[index]
        input_tensor = torch.LongTensor(input_ids)
        st_one_hot = torch.zeros_like(input_tensor, dtype=torch.long)
        st_one_hot[st] = 1
        ed_ont_hot = torch.zeros_like(input_tensor, dtype=torch.long)
        ed_ont_hot[ed] = 1

        return input_tensor, st_one_hot, ed_ont_hot

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    train_set = CMRCDataset('squad-style-data/cmrc2018_train.json', tokenizer)
    print(len(train_set))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=4,
    )
    print(next(iter(train_loader)))