import _pickle

class TeamDfWorkflow(object):
    def __init__(self):
        self.season_id = '22022'
        self.team_dfs_filename = 'batch_prediction/utils/recent_team_dfs'
        self.abbreviation_mapping={'HOU': {'team_name': 'Houston Rockets', 'team_id': 1610612745},
                                 'UTA': {'team_name': 'Utah Jazz', 'team_id': 1610612762},
                                 'SAS': {'team_name': 'San Antonio Spurs', 'team_id': 1610612759},
                                 'LAL': {'team_name': 'Los Angeles Lakers', 'team_id': 1610612747},
                                 'POR': {'team_name': 'Portland Trail Blazers', 'team_id': 1610612757},
                                 'GSW': {'team_name': 'Golden State Warriors', 'team_id': 1610612744},
                                 'MEM': {'team_name': 'Memphis Grizzlies', 'team_id': 1610612763},
                                 'CLE': {'team_name': 'Cleveland Cavaliers', 'team_id': 1610612739},
                                 'WAS': {'team_name': 'Washington Wizards', 'team_id': 1610612764},
                                 'MIL': {'team_name': 'Milwaukee Bucks', 'team_id': 1610612749},
                                 'TOR': {'team_name': 'Toronto Raptors', 'team_id': 1610612761},
                                 'SAC': {'team_name': 'Sacramento Kings', 'team_id': 1610612758},
                                 'DAL': {'team_name': 'Dallas Mavericks', 'team_id': 1610612742},
                                 'IND': {'team_name': 'Indiana Pacers', 'team_id': 1610612754},
                                 'ORL': {'team_name': 'Orlando Magic', 'team_id': 1610612753},
                                 'PHI': {'team_name': 'Philadelphia 76ers', 'team_id': 1610612755},
                                 'DEN': {'team_name': 'Denver Nuggets', 'team_id': 1610612743},
                                 'CHI': {'team_name': 'Chicago Bulls', 'team_id': 1610612741},
                                 'DET': {'team_name': 'Detroit Pistons', 'team_id': 1610612765},
                                 'PHX': {'team_name': 'Phoenix Suns', 'team_id': 1610612756},
                                 'MIA': {'team_name': 'Miami Heat', 'team_id': 1610612748},
                                 'ATL': {'team_name': 'Atlanta Hawks', 'team_id': 1610612737},
                                 'BOS': {'team_name': 'Boston Celtics', 'team_id': 1610612738},
                                 'NYK': {'team_name': 'New York Knicks', 'team_id': 1610612752},
                                 'CHA': {'team_name': 'Charlotte Hornets', 'team_id': 1610612766},
                                 'LAC': {'team_name': 'Los Angeles Clippers', 'team_id': 1610612746},
                                 'MIN': {'team_name': 'Minnesota Timberwolves', 'team_id': 1610612750},
                                 'OKC': {'team_name': 'Oklahoma City Thunder', 'team_id': 1610612760},
                                 'BKN': {'team_name': 'Brooklyn Nets', 'team_id': 1610612751},
                                 'NOP': {'team_name': 'New Orleans Pelicans', 'team_id': 1610612740}}
        self._get_dfs()
    
    def _get_dfs(self):
        with open(self.team_dfs_filename, 'rb') as f:
            self.dfs = _pickle.load(f)
        self.team_ids = list(self.dfs.keys())
        
    def _update_dfs(self):
        with open(self.team_dfs_filename, 'wb') as f:
            _pickle.dump(self.dfs, f)