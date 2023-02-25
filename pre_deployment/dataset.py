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



from sklearn.preprocessing import RobustScaler

class NBAPlayerData(Dataset):
    def __init__(self, df, scaler):
        self.stats = ['AST', 'BLK', 'DREB', 'FANTASY_PTS', 'FG3A', 'FG3M', 'FGA', 'FGM',
                      'FG_PCT', 'FTA', 'FTM', 'OREB', 'PF', 'PLUS_MINUS', 'PTS',
                      'REB', 'STL', 'TOV']
        #self.df = df.dropna().reset_index(drop=True)
        self.df = df
        self.scaler = scaler
        
        self._arrange_stats()
        self._arrange_momentum()
        self._arrange_odds()
        self._arrange_target()
        self._arrange_game_ids()
        
    def _arrange_stats(self):
        # NCHW: row, team, stat, player, past_game
        df_stats = self.df[[col for col in self.df.columns if col[:-9] in self.stats]]
        if self.scaler == 'fit':
            self.scaler = RobustScaler()
            df_stats = pd.DataFrame(self.scaler.fit_transform(df_stats), columns=df_stats.columns)
        else:
            df_stats = pd.DataFrame(self.scaler.transform(df_stats), columns=df_stats.columns)
            
        stat_names_all = []
        for team in ['_A', '_B']:
            stat_names_per_team = [col for col in self.df.columns if ((col[:-9] in self.stats) and (col[-2:] == team))]
            for stat in self.stats:
                stat_names = sorted([col for col in stat_names_per_team if col[:-9]==stat])
                stat_names_all.extend(stat_names)

        df_stats = df_stats[stat_names_all] # refactor
        stats_tensor = torch.tensor(df_stats.to_numpy().reshape(-1, 2, 18, 8, 8))
        self.X = stats_tensor

    def _arrange_momentum(self):
        # NCHW: row, team, past_game
        df_momentum = self.df[[col for col in self.df.columns if 'momentum' in col]]
        momentum_tensor = torch.tensor(df_momentum.astype('int').to_numpy().reshape(-1, 2, 8))
        self.momentum = momentum_tensor
        
    def _arrange_odds(self):
        self.odds = torch.tensor(self.df[['home_odds', 'away_odds']].astype('float').to_numpy())
        
    def _arrange_target(self):
        self.y = torch.tensor(self.df['TEAM_A_WIN'].astype('int').to_numpy())
        
    def _arrange_game_ids(self):
        self.game_ids = self.df['GAME_ID'].astype('int').to_numpy()

    # Define len function
    def __len__(self):
        return len(self.y)

    # Define getitem function
    def __getitem__(self, idx):
        stats = self.X[idx,:]
        momentum = self.momentum[idx,:]
        odds = self.odds[idx]
        target = self.y[idx]
        game_ids = self.game_ids[idx]
        
        return stats, momentum, odds, target, game_ids


# In[ ]:




