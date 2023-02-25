import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

from pytorch_lightning import LightningModule, Trainer, seed_everything
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


# In[ ]:


from model import PlayerLevelCNN
from dataset import NBAPlayerData


# In[ ]:


class CustomCrossValidation(object):
    def __init__(self, df, i=10):
        self.seasons = ['22008', '22009', '22010', '22011', '22012', '22013', '22014',
                        '22015', '22016', '22017', '22018', '22019', '22020', '22021', 
                        '22022']
        self.fold_slice = i
        
        self.df = df.copy()
    
    def _get_split(self, get_half):
        '''
        get half is for having observations that belong to the same season 
        both in train and validation sets, thereby to reduce the impact of
        a possible domain shift
        '''
        train_set = self._get_train_set(get_half)
        valid_set = self._get_valid_set(get_half)
        return train_set, valid_set
    
    def _get_train_set(self, get_half=False):
        res = self.df[self.df['SEASON_ID'].isin(self.seasons[:self.fold_slice])]
        if get_half:
            next_season = self.df[self.df['SEASON_ID']==self.seasons[self.fold_slice]].reset_index(drop=True)
            res = pd.concat([res, next_season.iloc[:len(next_season)//2]])
        return res
    
    def _get_valid_set(self, get_half=False):
        res = self.df[self.df['SEASON_ID']==self.seasons[self.fold_slice]]
        if get_half:
            res = res.reset_index(drop=True).iloc[len(res)//2:]
        return res
    
    def _slide(self):
        if self.fold_slice + 1 < len(self.seasons):
            self.fold_slice += 1
        else:
            return True
        
    def _prepare_data_loaders(self, get_half):
        BATCH_SIZE = 256
        NUM_WORKERS = 0
        
        df_tr, df_val = self._get_split(get_half)
        print(f"train set range {df_tr['GAME_DATE'].min()} - {df_tr['GAME_DATE'].max()}")
        print(f"valid set range {df_val['GAME_DATE'].min()} - {df_val['GAME_DATE'].max()}")
            
        train_set = NBAPlayerData(df_tr, scaler='fit')
        train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        val_set = NBAPlayerData(df_val, scaler=train_set.scaler)
        val_loader = DataLoader(val_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        data_loaders = [train_loader, val_loader]
        return data_loaders
        
    def _fit_model(self, model_params, data_loaders):
        model = PlayerLevelCNN(model_id='trial_model', data_loaders=data_loaders, hyperparams=model_params, seed=30)
        
        checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                              dirpath="cv_logs/",
                                              save_top_k=1,
                                              mode="min",
                                              every_n_epochs=1)

        early_stopping = EarlyStopping('val_loss', patience = 5, mode='min') 

        trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,  
            max_epochs=5,
            callbacks=[checkpoint_callback, early_stopping],
            logger=CSVLogger(save_dir="cv_logs/"),
            deterministic=True
        )
        trainer.fit(model)
        result = trainer.validate(model)
        return result, model
        
    def _predict_val_set(self, model, val_loader):
        predictions = {}
        
        val_loader_iter = iter(val_loader)
        for i in range(len(val_loader)):
            stats, momentum, odds, target, game_ids = next(val_loader_iter)
            batch_preds = model.predict_step(stats, momentum, odds, predict_label=True)
            predictions.update({'00'+str(int(game_ids[i])):bool(batch_preds[i]) for i in range(len(batch_preds))})
        
        return predictions

    def _get_updated_df(self):
        # update the duplicated game_ids
        keys_to_change = []
        for k, v in self.all_predictions.items():
            if len(k) > 10:
                keys_to_change.append(k)
        for k in keys_to_change:
            self.all_predictions[k[:-1] + '_2'] = self.all_predictions[k]
            self.all_predictions.pop(k)

        predicted_df = self.df.copy()
        predicted_df['cv_pred'] = predicted_df['GAME_ID'].map(self.all_predictions)
        predicted_df.dropna(subset='cv_pred', inplace=True)
        predicted_df = predicted_df[['GAME_ID', 'GAME_DATE', 'SEASON_ID', 'home_team',
                                     'away_team', 'TEAM_A_WIN', 'cv_pred']]
        self.predicted_df = predicted_df
            
    def fit_predict(self, model_params, get_half):
        
        self.all_predictions = {}
        self.fold_scores = []
        j = 0
        while True:
            print(j)
            data_loaders = self._prepare_data_loaders(get_half)
            
            result, model = self._fit_model(model_params, data_loaders)
            self.fold_scores.append(result)
            
            predictions = self._predict_val_set(model, data_loaders[1])
            self.all_predictions.update(predictions)
            
            if self._slide():
                break
            j += 1
            print()
        
        self._get_updated_df()
        return self.predicted_df

