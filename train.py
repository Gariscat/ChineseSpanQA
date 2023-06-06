from model import BertQA
from data import *
import pytorch_lightning as pl
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pretrained_path', type=str, default='hfl/chinese-bert-wwm-ext')
    parser.add_argument('--pretrained_path', type=str, default='bert-base-chinese')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)

    train_set = CMRCDataset('squad-style-data/cmrc2018_train.json', tokenizer)
    val_set = CMRCDataset('squad-style-data/cmrc2018_dev.json', tokenizer)
    train_loader = DataLoader(dataset=train_set, batch_size=4,)
    val_loader = DataLoader(dataset=val_set, batch_size=4,)

    model = BertQA(args.pretrained_path)
    trainer = pl.Trainer(auto_select_gpus=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)