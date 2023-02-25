import numpy as np
import pandas as pd
import _pickle
import warnings
from nba_api.stats.endpoints.scoreboardv2 import ScoreboardV2

warnings.filterwarnings("ignore")

from .team_data_workflow import TeamDfWorkflow
from .odds_scraper import OddsScraper

warnings.filterwarnings("ignore")

class PreGameWorkflow(TeamDfWorkflow):
    def __init__(self):
        super().__init__()
        self.cols_filename = 'batch_prediction/utils/columns_file'
        self._get_cols()
        self.team_id_mapping = {v['team_id']:{'team_name':v['team_name'], 'abbreviation':k} 
                                for k, v in self.abbreviation_mapping.items()}
        
    def _get_cols(self):
        with open(self.cols_filename, 'rb') as f:
            columns = _pickle.load(f)
        self.export_columns = columns['export_columns']
        self.meta_data_columns = columns['meta_data_columns']
        
    def _get_games(self):
        scoreboard_dict = ScoreboardV2().get_dict()
        games = {'game_date': scoreboard_dict['parameters']['GameDate'], 
                 'games': {}}
        
        game_info_list = scoreboard_dict['resultSets'][0]['rowSet']
        for game_info in game_info_list:
            game_id = game_info[2]
            home_team_id = game_info[6]
            home_team_name, home_team_abbr = self.team_id_mapping[home_team_id]['team_name'], self.team_id_mapping[home_team_id]['abbreviation']
            away_team_id = game_info[7]
            away_team_name, away_team_abbr = self.team_id_mapping[away_team_id]['team_name'], self.team_id_mapping[away_team_id]['abbreviation']
            games['games'][game_id] = {'home_team': {'team_name':home_team_name, 'team_id':home_team_id, 'abbreviation':home_team_abbr},
                                       'away_team': {'team_name':away_team_name, 'team_id':away_team_id, 'abbreviation':away_team_abbr}}
        self.games = games
        
    def _shift_df(self, team_id):
        df = self.dfs[team_id]
        feature_cols = [col for col in df.columns if '_t0' in col]
        shifted = [df[feature_cols].shift(-t).loc[:0].rename(columns={col:col.replace('_t0', f'_t{t}') for col in df.columns})
                  for t in range(1,9)]
        return pd.concat(shifted, axis=1)
            
    def _prepare_half_row(self, team_id, home_away):
        df = self._shift_df(team_id)
        
        row = dict(df.loc[0, self.export_columns])
        if home_away == 'home':
            new_keys = [f'{k}_A' for k in row.keys()]
        elif home_away == 'away':
            new_keys = [f'{k}_B' for k in row.keys()]
        else:
            raise Exception('home_away should be either home or away')
        row = dict(zip(new_keys, list(row.values())))
        
        return row
    
    def _add_odds(self):
        required_games = set()
        for i, row in self.new_games.iterrows():
            game_key = row['home_team'] + ' vs. ' + row['away_team']
            required_games.add(game_key)
        
        scraper = OddsScraper(required_games)
        odds = scraper.get_odds()
        
        self.new_games['home_odds'] = None
        self.new_games['away_odds'] = None
        for i, row in self.new_games.iterrows():
            game_key = row['home_team'] + ' vs. ' + row['away_team']
            game_odds = odds[game_key]
            self.new_games.at[i, 'home_odds'] = game_odds['home_odds']
            self.new_games.at[i, 'away_odds'] = game_odds['away_odds']
        
    def prepare_new_games(self):
        self._get_games()
        rows = []
        game_date = self.games['game_date']
        for game_id, game_info in self.games['games'].items():
            home_id, away_id = game_info['home_team']['team_id'], game_info['away_team']['team_id']
            
            row = self._prepare_half_row(home_id, 'home')
            row_away = self._prepare_half_row(away_id, 'away')
            row.update(row_away)

            row['GAME_ID'], row['GAME_DATE'] = game_id, game_date
            row['SEASON_ID'] = self.season_id

            home_team, away_team = game_info['home_team']['team_name'], game_info['away_team']['team_name']
            row['home_team'], row['away_team'] = home_team, away_team

            home_id, away_id = game_info['home_team']['team_id'], game_info['away_team']['team_id']
            row['TEAM_ID_A'], row['TEAM_ID_B'] = home_id, away_id

            home_abbr, away_abbr = game_info['home_team']['abbreviation'], game_info['away_team']['abbreviation']
            home_matchup = f'{home_abbr} vs. {away_abbr}'
            away_matchup = home_matchup.replace('vs.', '@')
            row['MATCHUP_A'], row['MATCHUP_B'] = home_matchup, away_matchup

            row['WL_A'], row['WL_B'], row['TEAM_A_WIN'] = None, None, None
            
            rows.append(row)
        if rows != []:
            self.new_games = pd.DataFrame(rows)
            self._add_odds() # to be implemented
        else:
            self.new_games = None
        return self.new_games
