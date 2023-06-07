from model import *
import argparse
from transformers import BertTokenizer, AutoTokenizer
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='......')
    parser.add_argument('--pretrained_path', type=str, default='/root/chinese-bert-wwm-ext')
    args = parser.parse_args()
    """
    args.ckpt_path = '...'
    print(args.ckpt_path)
    exit()
    """
    tokenizer_cls = BertTokenizer if 'bert' in args.pretrained_path else AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    model = ModelQA.load_from_checkpoint(args.ckpt_path, pretrained_path=args.pretrained_path)
    model.eval()

    while True:
        context = input('段落：')
        if len(context) == 0:
            break
        question = input('问题：')

        input_ids = tokenizer.encode(
            text=context,
            text_pair=question,
            padding='max_length',
            max_length=512,
        )
        input_tensor = torch.LongTensor(input_ids).reshape(1, -1)
        st_scores, ed_scores = model(input_tensor=input_tensor)
        st_pred, ed_pred = torch.argmax(st_scores).item(), torch.argmax(ed_scores).item()
        answer = tokenizer.decode(token_ids=input_ids[st_pred:ed_pred+1])
        print(answer)
        
    