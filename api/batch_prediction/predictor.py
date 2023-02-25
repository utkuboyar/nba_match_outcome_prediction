import pandas as pd
import numpy as np
import _pickle
from scipy.special import softmax
import datetime
import onnxruntime

class Predictor(object):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self._prepare_data()
        self._predict()
        
    def _predict(self):
        filepath = 'batch_prediction/models/nba_net10.onnx'
        ort_session = onnxruntime.InferenceSession(filepath)
        
        input_name0 = ort_session.get_inputs()[0].name
        input_name1 = ort_session.get_inputs()[1].name
        input_name2 = ort_session.get_inputs()[2].name # new
        
        predictions = {}
        for i in self.df.index:
            game_id, home_team, away_team, date = self.metadata.loc[i, ['GAME_ID', 'home_team', 'away_team', 'GAME_DATE']]
            
            stats = self.df_stats.loc[i].to_numpy(dtype=np.float64).reshape(1, 2, 18, 8, 8)
            momentum = self.df_momentum.loc[i].to_numpy(dtype=np.int32).reshape(1, 2, 8)
            odds = self.df_odds.loc[i].to_numpy(dtype=np.float64).reshape(1, 2)
            
            ort_inputs = {input_name0: stats, 
                          input_name1: momentum,
                          input_name2: odds}
            logits = ort_session.run(None, ort_inputs)[0]
            probs = softmax(logits)
            
            home_prob, away_prob = probs[0,1].round(2), probs[0,0].round(2)
            if home_prob >= away_prob:
                label = 'home'
            else:
                label = 'away'
            predictions[game_id] = {'meta':{'game_date':date, 'home_team':home_team, 'away_team':away_team},
                                    'predictions':{'label': label,
                                                   'status': 'undetermined',
                                                   'probability': {'home': home_prob, 'away':away_prob},
                                                   'odds': {'home':np.round(1/home_prob, 3), 'away':np.round(1/away_prob, 3)}}}
        json = {}
        now = datetime.datetime.now()
        json['prediction_date'] = f'{now.year}-{now.month}-{now.day}'
        json['games'] = predictions
        self.predictions = json
        
    def _prepare_data(self):
        self._arrange_stats()
        self._arrange_momentum()
        self._arrange_odds()
        self._arrange_metadata()
        
    def _arrange_stats(self):
        self.stats = ['AST', 'BLK', 'DREB', 'FANTASY_PTS', 'FG3A', 'FG3M', 'FGA', 'FGM',
                      'FG_PCT', 'FTA', 'FTM', 'OREB', 'PF', 'PLUS_MINUS', 'PTS',
                      'REB', 'STL', 'TOV']
        stat_names_all = []
        for team in ['_A', '_B']:
            stat_names_per_team = [col for col in self.df.columns if ((col[:-9] in self.stats) and (col[-2:] == team))]
            for stat in self.stats:
                stat_names = sorted([col for col in stat_names_per_team if col[:-9]==stat])
                stat_names_all.extend(stat_names)
        df_stats = self.df[stat_names_all] # refactor
        
        with open('batch_prediction/utils/robust_scaler', 'rb') as f:
            self.scaler = _pickle.load(f)
        df_stats = pd.DataFrame(self.scaler.transform(df_stats), columns=df_stats.columns)
        
        self.df_stats = df_stats

    def _arrange_momentum(self):
        # NCHW: row, team, past_game
        self.df_momentum = self.df[[col for col in self.df.columns if 'momentum' in col]]
        
    def _arrange_odds(self):
        self.df_odds = self.df[['home_odds', 'away_odds']]
        
    def _arrange_metadata(self):
        with open('batch_prediction/utils/columns_file', 'rb') as f:
            self.metadata_cols = _pickle.load(f)['meta_data_columns']
        self.metadata = self.df[self.metadata_cols]
        
    def get_predictions(self):
        return self.predictions