from model import BertQA
from data import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='/root/chinese-bert-wwm-ext')
    parser.add_argument('--opt_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=10)
    # parser.add_argument('--pretrained_path', type=str, default='bert-base-chinese')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)

    max_length = 512 if 'xlnet' not in args.pretrained_path else None
    train_set = CMRCDataset('squad-style-data/cmrc2018_train.json', tokenizer, max_length=max_length)
    val_set = CMRCDataset('squad-style-data/cmrc2018_dev.json', tokenizer, max_length=max_length)
    train_loader = DataLoader(dataset=train_set, batch_size=4,)
    val_loader = DataLoader(dataset=val_set, batch_size=4,)

    model = BertQA(args.pretrained_path, opt_name=args.opt_name, lr=args.lr)
    wandb_logger = WandbLogger(
        entity='gariscat',
        project='ChineseSpanQA',
        config={
            'pretrained_path': args.pretrained_path,
            'opt_name': args.opt_name,
            'lr': args.lr,
            'max_epochs': args.max_epochs,
        },
        log_model=True
    )
    trainer = pl.Trainer(accelerator="gpu", devices="auto", logger=wandb_logger, max_epochs=args.max_epochs, deterministic=True)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
