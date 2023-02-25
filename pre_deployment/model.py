import torch
from torch import optim
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

import torchmetrics
from torchmetrics import Accuracy

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import _pickle
import os

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')
print('device: ', device)



class PlayerLevelCNN(LightningModule):
    def __init__(self, model_id, data_loaders, hyperparams, seed=None):
        super().__init__()
        
        learning_rate = hyperparams['learning_rate'] 
        loss_func = hyperparams['loss_func']
        L1 = hyperparams['L1']
        c = hyperparams['c']
        activation = hyperparams['activation']
        hidden_layer_dropout = hyperparams['hidden_layer_dropout'] 
        input_dropout = hyperparams['input_dropout'] 
        out_channels1 = hyperparams['out_channels1'] 
        out_channels2 = hyperparams['out_channels2']
        
        if seed is not None:
            self.seed = seed
            seed_everything(seed)
            
        self.model_id = model_id

        self.data_loader_train, self.data_loader_val = data_loaders
        
        x, momentum, odds, _, _ = next(iter(self.data_loader_train))
        input_eg = (x[0].reshape(1, 2, 18, 8, 8), momentum[0].reshape(1, 2, 8), odds[0].reshape(1, 2))
        
        #self.example_input_array = (torch.rand(4, 2, 18, 8, 8), torch.rand(4, 2, 8))
        self.example_input_array = input_eg
        
        self.learning_rate = learning_rate
        self.L1 = L1
        self.act = activation
        exec(f"self.act_fn = F.{activation}")
        
        if loss_func == 'cross_entropy':
            self.loss_fn_name = 'CE'
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_func == 'sample_weighted_cross_entropy':
            self.loss_fn_name = 'SWCE'
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        elif loss_func == 'odds_decorrelation':
            self.loss_fn_name = 'OD'
            self.loss_fn = OddsDecorrelation(c=c)
            self.c = c
            
        _, _, self.stats_count, self.player_count, self.game_count = next(iter(self.data_loader_train))[0].shape
        
        self.test_accuracy = Accuracy(task='binary')
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        
        self.C1 = nn.Conv3d(in_channels=2, out_channels=out_channels1, kernel_size=(1, 1, self.game_count), 
                            groups=2, bias=False)
        self.C2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=(1, self.player_count), 
                                    groups=2, bias=False)
        
        self.Cm = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, self.game_count))
        
        self.D64 = nn.Linear(out_channels2*self.stats_count + 2 + 1, 64, bias=False)
        self.D16 = nn.Linear(64, 16, bias=False)
        self.D2 = nn.Linear(16, 2)
        
        self.C1_bn = nn.BatchNorm3d(out_channels1)
        self.C2_bn = nn.BatchNorm2d(out_channels2)
        self.D64_bn = nn.BatchNorm1d(64)
        self.D16_bn = nn.BatchNorm1d(16)
        
        self.input_dropout = nn.Dropout(input_dropout)
        self.D64_dropout = nn.Dropout(hidden_layer_dropout)
        self.D16_dropout = nn.Dropout(hidden_layer_dropout)
        
        
    def forward(self, *args):
        x, momentum, odds = args
        batch_size = x.shape[0]
        
        x = self.C1(x.float())
        x = self.C1_bn(x)
        x = self.act_fn(x)
        x = x.reshape(batch_size, -1, self.stats_count, self.player_count)
        
        x = self.C2(x)
        x = self.C2_bn(x)
        x = self.act_fn(x)
        
        momentum = momentum.reshape(batch_size, 1, 2, -1)
        momentum = self.Cm(momentum.float())
        momentum = momentum.view(momentum.shape[0], -1)
        
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, momentum, odds[:,0].float().reshape(batch_size, 1)), 1) 
        x = self.input_dropout(x)
        
        x = self.D64(x)
        x = self.D64_bn(x)
        x = self.act_fn(x)
        x = self.D64_dropout(x)

        x = self.D16(x)
        x = self.D16_bn(x)
        x = self.act_fn(x)
        x = self.D16_dropout(x)

        x = self.D2(x)

        return x
    
    def _get_preds_and_losses(self, x, momentum, odds, y):
        #x, momentum = model_input
        logits = self(x, momentum, odds)
        y_hat = F.softmax(logits)
        
        if self.loss_fn_name == 'CE':
            loss = self.loss_fn(logits, y.type(torch.LongTensor).cuda())
        elif self.loss_fn_name == 'SWCE':
            loss = self.loss_fn(logits, y) * bookmaker_residuals
            loss = loss.mean()
        elif self.loss_fn_name == 'OD':
            loss = self.loss_fn(logits, y, odds)
            
        preds = logits.argmax(1)
        return loss, preds
        
    def training_step(self, batch, batch_idx):
        x, momentum, odds, y, _ = batch
        #model_input = (x, momentum)
        loss, preds = self._get_preds_and_losses(x, momentum, odds, y)
        self.train_accuracy.update(preds, y)
        
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_accuracy, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, momentum, odds, y, _ = batch
       #model_input = (x, momentum)
        loss, preds = self._get_preds_and_losses(x, momentum, odds, y)
        self.val_accuracy.update(preds, y)

        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, prog_bar=False, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x, momentum, odds, y, _ = batch
        #model_input = (x, momentum)
        loss, preds = self._get_preds_and_losses(x, momentum, odds, y)
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss, prog_bar=False)
        self.log("test_acc", self.test_accuracy, prog_bar=False)
        
    def predict_step(self, x, momentum, odds, predict_label=False, threshold=0.5):
        logits = self(x, momentum, odds)
        predicted_prob = F.softmax(logits)[:,1]
        if predict_label:
            return predicted_prob >= threshold
        return predicted_prob

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.L1)
        return optimizer
    
    def train_dataloader(self):
        return self.data_loader_train
    
    def val_dataloader(self):
        return self.data_loader_val
    
    def get_model_id(self):
        return self.model_id


# In[ ]:




