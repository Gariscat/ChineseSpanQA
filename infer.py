from model import *
import argparse
from transformers import BertTokenizer, AutoTokenizer
import torch

NULL_PRINT = '抱歉，关于这个问题我暂时没有可以提供的信息^_^'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='ckpt/epoch=9-step=14210.ckpt')
    parser.add_argument('--pretrained_path', type=str, default='hfl/chinese-bert-wwm-ext')
    args = parser.parse_args()
    
    tokenizer_cls = BertTokenizer if 'bert' in args.pretrained_path else AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    model = ModelQA.load_from_checkpoint(args.ckpt_path, pretrained_path=args.pretrained_path)
    model.eval()
    context = input('背景：')
    
    while True:
        if len(context) == 0:
            break
        question = input('问题：')
        
        input_ids_a = tokenizer.encode(
            text=context,
            padding='max_length',
            max_length=480,
            truncation=True,
        )
        input_ids_b = tokenizer.encode(
            text=question,
            padding='max_length',
            max_length=32,
            truncation=True,
        )
        input_ids = input_ids_a + input_ids_b
        # assert len(input_ids) <= 512
        input_tensor = torch.LongTensor(input_ids).reshape(1, -1)
        st_scores, ed_scores = model(input_tensor=input_tensor)
        st_pred, ed_pred = torch.argmax(st_scores).item(), torch.argmax(ed_scores).item()
        # print(st_pred, ed_pred)
        if ed_pred > st_pred:
            answer = tokenizer.decode(token_ids=input_ids[st_pred:ed_pred+1])
            answer = answer.replace(' ', '')
            print(answer)
        else:
            print(NULL_PRINT)