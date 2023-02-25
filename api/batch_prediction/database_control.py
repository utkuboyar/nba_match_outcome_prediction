import _pickle
from bson import ObjectId

from betting_optimization.portfolio import Portfolio
from .utils.db_connection import get_collection

class DatabaseController(object):
    def __init__(self):
        self.games_collection = get_collection('games')
        self.rounds_collection = get_collection('rounds')

    def add_predictions(self, preds, odds):
        prediction_date = preds['prediction_date']
        games = preds['games']

        ratios, decisions = self._get_bet_ratios(preds, odds)

        game_round = {"prediction_date": prediction_date,
                      "rule_in":decisions['rules'],
                      'values':{'n_played':str(decisions['n_played']), 'lower_limit':str(decisions['lower_lim'])}}
        print(game_round)
        self.rounds_collection.insert_one(game_round)
        
        for game_id, game_info in games.items():
            betting_ratio, bookmaker_odds = ratios[game_id]
            predicted_game = {
                "game_id": game_id,
                "home_team": game_info['meta']['home_team'], "away_team": game_info['meta']['away_team'],
                "game_date":game_info['meta']['game_date'], "prediction_date": prediction_date,
                "predicted_label":game_info['predictions']['label'], "status": game_info['predictions']['status'], 
                "predicted_proba_home":str(game_info['predictions']['probability']['home']), "predicted_proba_away":str(game_info['predictions']['probability']['away']),
                "predicted_odds_home": str(game_info['predictions']['odds']['home']), "predicted_odds_away":str(game_info['predictions']['odds']['away']),
                "bet_ratio":betting_ratio, 'bookmaker_odds':bookmaker_odds, 
                "bookmaker_odds_home":odds.at[game_id, 'home_odds'], "bookmaker_odds_away":odds.at[game_id, 'away_odds']}
            # print(predicted_game)
                 
            game = self.games_collection.find_one({'game_id':game_id}, {"_id":1, 'game_id': 1, "status": 1, "predicted_label": 1})
            if game:
                continue

            self.games_collection.insert_one(predicted_game)

    def _get_bet_ratios(self, preds, odds):
        game_ids, probs, given_odds = [], [], []
        for game_id, game_info in preds['games'].items():
            game_ids.append(game_id)
            bet_for = game_info['predictions']['label']
            probs.append(float(game_info['predictions']['probability'][bet_for]))
            given_odds.append(float(odds.at[game_id, f'{bet_for}_odds']))

        portfolio = Portfolio(probs=probs, odds=given_odds)
        ratios, decisions = portfolio.run(game_ids)

        return ratios, decisions

    def update_prediction_results(self, updated_games):

        for game_id, teams in updated_games.items():
            if len(teams) > 2:
                raise Exception('problem: DatabaseController.update_prediction_results')

            # check if there already is a record for the game
            game = self.games_collection.find_one({'game_id':game_id}, {"_id":1, "status": 1, "predicted_label": 1})
            # print('here1')
            if not game:
                continue
            # print(game['status'])
            if game['status'] == 'undetermined':
                outcome = self._check_game_outcome(game_id, teams)
                pred = game['predicted_label']
                # print('here3:', outcome, pred)

                if pred == outcome:
                    game['status'] = 'correct'
                else:
                    game['status'] = 'wrong'

                self.games_collection.update_one({'_id':ObjectId(game['_id'])}, {"$set": game}) 

    def _check_game_outcome(self, game_id, team_ids):
        # print('here2')
        with open('batch_prediction/utils/recent_team_dfs', 'rb') as f:
            dfs = _pickle.load(f)
        team_df1 = dfs[team_ids[0]].set_index('GAME_ID')
        team_name1 = team_df1.at[game_id, 'team_name']
        wl1 = team_df1.at[game_id, 'momentum_t0']

        team_df2 = dfs[team_ids[1]].set_index('GAME_ID')
        team_name2 = team_df2.at[game_id, 'team_name']
        wl2 = team_df2.at[game_id, 'momentum_t0']

        if wl1 == wl2:
            print(team_name1, wl1, team_name2, wl2)
            raise Exception('problem 1: DatabaseController._check_game_outcome')
        if wl1:
            winner = team_name1
        else:
            winner = team_name2

        game = self.games_collection.find_one({'game_id':game_id}, {"_id":1, "home_team": 1, "away_team":1})
        if game['home_team'] == winner:
            outcome = 'home'
        elif game['away_team'] == winner:
            outcome = 'away'
        else:
            raise Exception('problem 2: DatabaseController._check_game_outcome')
        return outcome