import numpy as np
import pandas as pd
import _pickle
import datetime
import warnings
from nba_api.stats.endpoints import leaguegamelog

from .team_data_workflow import TeamDfWorkflow

warnings.filterwarnings("ignore")

class PostGameWorkflow(TeamDfWorkflow):
    def __init__(self, db_controller):
        super().__init__()
        self.seasons = ['2022-23']
        self.updated_teams = []
        self.updated_games = {}
        
        self.db_controller = db_controller
        #self.last_predictions_id, self.last_predictions = self.model_tracker.get_last_prediction()
        
    def _gather_raw_data(self):
        season = self.seasons[0]
        player_league_info = leaguegamelog.LeagueGameLog(player_or_team_abbreviation="P", season=season)
        player_info = player_league_info.get_data_frames()
        raw_data = player_info[0]
        raw_data['GAME_DATE'] = pd.to_datetime(raw_data['GAME_DATE'], format='%Y-%m-%d')
        raw_data.sort_values('GAME_DATE', ascending=False, inplace=True)
        self.raw_data = raw_data
        self.update_time = datetime.datetime.now()
        
    def _groupby_team(self):
        self.raw_team_dfs = self.raw_data.groupby('TEAM_ID')
        
    def _dump(self):
        ####
        if self.updated_teams == []:
            return
        ####
        with open(self.team_dfs_filename, 'wb') as f:
            _pickle.dump(self.dfs, f)
            
        #self.model_tracker.update_prediction_results(self.last_predictions_id, self.last_predictions)
        self.db_controller.update_prediction_results(self.updated_games)
        
    

    def _extend_updated_games(self, team_id, game_ids):
        for game_id in game_ids:
            if game_id in self.updated_games:
                self.updated_games[game_id].append(team_id)
            else:
                self.updated_games[game_id] = [team_id]

    def update_recent_dfs(self):
        self._gather_raw_data()
        self._groupby_team()
        for team_id in self.team_ids:
            raw_team_df = self.raw_team_dfs.get_group(team_id)
            recent_team_df = self.dfs[team_id]
            team_updater = TeamDataUpdater(team_id=team_id, raw_team_df=raw_team_df, 
                                           recent_df=recent_team_df, season_id=self.season_id)
            updated_team_df = team_updater._process()
            if not recent_team_df.equals(updated_team_df):
                #print('here: update_recent_dfs')
                self._extend_updated_games(team_id, team_updater.games_to_record)
                #self._update_prediction_result(updated_team_df)
                self.updated_teams.append(team_id)
                self.dfs[team_id] = updated_team_df
        self._dump()


class TeamDataUpdater(object):
    def __init__(self, team_id, raw_team_df, recent_df, season_id):
        self.team_id = team_id
        self.team_df = raw_team_df
        self.recent_df = recent_df
        self.season_id = season_id
        
        self.stats = ['AST', 'BLK', 'DREB', 'FANTASY_PTS', 'FG3A', 'FG3M', 'FGA', 'FGM',
                      'FG_PCT', 'FTA', 'FTM', 'OREB', 'PF', 'PLUS_MINUS', 'PTS',
                      'REB', 'STL', 'TOV']
        self.player_count = 8
        self.prev_game_count = 8
        self.games_processed = []
        
        self._sortby_time()
        self._groupby_game()
        
    def _sortby_time(self):
        self.team_df.sort_values('GAME_DATE', ascending=False, inplace=True)
        
    def _groupby_game(self):
        self.game_ids = self.team_df['GAME_ID'].unique()
        self.game_dfs = self.team_df.groupby('GAME_ID')
        
    def _prepare_game_df(self, game_id):
        game = self.game_dfs.get_group(game_id)

        #game.set_index('PLAYER_ID', drop=True, inplace=True)
        game.sort_values('MIN', ascending=False, inplace=True)

        if len(game) < self.player_count:
            #print('here', len(game))
            missing = self.player_count - len(game)
            filler_df = pd.DataFrame({col:[None for i in range(missing)] for col in game.columns})
            game = pd.concat([game, filler_df])

        game.reset_index(drop=True, inplace=True)    
        return game.iloc[:self.player_count]

    def _process_game_stats(self, game_id):
        game = self._prepare_game_df(game_id)

        game.loc[((game['MIN'].isna()) | (game['MIN']==0)), 'MIN'] = 1
        game_stats = game[self.stats + ['MIN']]
        for stat in self.stats:
            game_stats[stat] /= game_stats['MIN']
        game_stats.drop(['MIN'], axis=1, inplace=True)

        player_ranks = np.array([i for i in range(1, self.player_count+1)] * len(self.stats))

        game_stats = game_stats.melt()

        game_stats['player_rank'] = player_ranks
        game_stats['player_rank'] = game_stats['player_rank'].astype('str')
        game_stats['variable'] = game_stats['variable'] + '_pl' + game_stats['player_rank'] + '_t0'
        game_stats.drop(['player_rank'], axis=1, inplace=True)

        game_stats.set_index('variable', drop=True, inplace=True)
        game_stats = game_stats.transpose()

        game_stats['GAME_DATE'] = game.at[0, 'GAME_DATE']
        game_stats['WL'] = game.at[0, 'WL']

        game_stats.index = [game_id]
        
        self.games_processed.append(game_stats)
        
    def _find_nonrecorded_games(self):
        last_game_recorded = self.recent_df.at[1, 'GAME_ID']
        
        i, self.found = 0, False
        while i <= len(self.game_ids)-1:
            if self.game_ids[i] == last_game_recorded:
                if i != 0:
                    self.found = True
                break
            i += 1
 
        games_to_record = list(self.game_ids[:i])
        games_to_record.reverse()
        self.games_to_record = games_to_record
        
    def _finalize_processed_games(self):
        self.games_processed = pd.concat(self.games_processed)
        self.games_processed['SEASON_ID'] = self.season_id
        self.games_processed['TEAM_ID'] = self.team_id
        self.games_processed['team_name'] = self.recent_df.at[0, 'team_name']
        self.games_processed.fillna(0, inplace=True)
        self.games_processed = self.games_processed.reset_index()
        self.games_processed['WL'] = self.games_processed['WL'] == 'W'
        self.games_processed.rename(columns={'index':'GAME_ID', 'WL':'momentum_t0'}, inplace=True)
        
    def _update_recent(self):
        most_recent_df = pd.concat([self.games_processed, self.recent_df])
        most_recent_df.drop_duplicates(subset=['GAME_ID'], keep='first', inplace=True)
        most_recent_df.dropna(subset=['GAME_ID'], inplace=True)
        row = self._prep_last_row()
        most_recent_df.sort_values('GAME_DATE', ascending=False, inplace=True)
        most_recent_df = pd.concat([row, most_recent_df])
        most_recent_df = most_recent_df.reset_index(drop=True)
        self.most_recent_df = most_recent_df.head(10)
        
    def _prep_last_row(self):
        row = {col:None for col in self.recent_df.columns}
        row['SEASON_ID'] = self.season_id
        row['TEAM_ID'] = self.team_id
        row['team_name'] = self.recent_df.at[0, 'team_name']
        return pd.DataFrame(row, index=range(1))
            
    def _process(self):
        self._find_nonrecorded_games()
        if not self.found:
            return self.recent_df
        for game_id in self.games_to_record:
            self._process_game_stats(game_id)
        self._finalize_processed_games()
        self._update_recent()
        return self.most_recent_df