from typing import Optional
from transformers import BertModel
from torch import nn, optim
import pytorch_lightning as pl

class BertQA(pl.LightningModule):
    def __init__(self, pretrained_path, opt_name='Adam', lr=1e-3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.st_out = nn.Linear(self.bert.config.hidden_size, 1)
        self.ed_out = nn.Linear(self.bert.config.hidden_size, 1)
        self.lr = lr
        self.opt_name = opt_name

    def forward(self, input_tensor):
        bert_outputs = self.bert(input_ids=input_tensor)
        st_scores = self.st_out(bert_outputs.last_hidden_state).squeeze(-1)
        ed_scores = self.ed_out(bert_outputs.last_hidden_state).squeeze(-1)
        return st_scores, ed_scores
    
    def configure_optimizers(self):
        opt_cls = optim.Adam if self.opt_name == 'Adam' else optim.SGD
        return opt_cls(self.parameters(), lr=self.lr)
    
    def training_step(self, train_batch, *args, **kwargs):
        input_tensor, st_ground_truth, ed_ground_truth = train_batch
        st_ground_truth = st_ground_truth.squeeze(-1)
        ed_ground_truth = ed_ground_truth.squeeze(-1)

        st_scores, ed_scores = self.forward(input_tensor)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(st_scores, st_ground_truth) + loss_func(ed_scores, ed_ground_truth)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, *args, **kwargs):
        input_tensor, st_ground_truth, ed_ground_truth = val_batch
        st_ground_truth = st_ground_truth.squeeze(-1)
        ed_ground_truth = ed_ground_truth.squeeze(-1)
        
        st_scores, ed_scores = self.forward(input_tensor)
        loss_func = nn.CrossEntropyLoss()
        """print('\n\n\n')
        print(st_scores.shape)
        print(st_ground_truth.shape)
        print('\n\n\n')"""
        loss = loss_func(st_scores, st_ground_truth) + loss_func(ed_scores, ed_ground_truth)
        self.log('val_loss', loss)