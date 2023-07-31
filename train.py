from model import ModelQA
from data import *
from transformers import BertTokenizer, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='/root/chinese-bert-wwm-ext')
    parser.add_argument('--opt_name', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--drop_p', type=float, default=0.5)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='/root/autodl-tmp')
    # parser.add_argument('--pretrained_path', type=str, default='bert-base-chinese')
    args = parser.parse_args()
    print('config:', vars(args))

    tokenizer_cls = BertTokenizer if 'bert' in args.pretrained_path else AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    # max_length = 512 if 'xlnet' not in args.pretrained_path else 1024
    max_length = 512
    train_set = CMRCDataset('squad-style-data/cleanned_cmrc2018_train.json', tokenizer, max_length=max_length)
    val_set = CMRCDataset('squad-style-data/cleanned_cmrc2018_dev.json', tokenizer, max_length=max_length)
    train_loader = DataLoader(dataset=train_set, batch_size=4,)
    val_loader = DataLoader(dataset=val_set, batch_size=4,)

    model = ModelQA(
        args.pretrained_path,
        opt_name=args.opt_name,
        lr=args.lr,
        drop_p=args.drop_p,
    )
    wandb_logger = WandbLogger(
        entity='gariscat',
        project='ChineseSpanQAPrototype',
        config={
            'pretrained_path': args.pretrained_path,
            'opt_name': args.opt_name,
            'lr': args.lr,
            'max_epochs': args.max_epochs,
            'drop_p': args.drop_p,
        },
        log_model=True,
        save_dir=args.log_dir,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        logger=wandb_logger,
        max_epochs=args.max_epochs,
        deterministic=True,
        default_root_dir=args.log_dir,
        log_every_n_steps=100,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
