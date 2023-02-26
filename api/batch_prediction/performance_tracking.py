import pandas as pd
import matplotlib.pyplot as plt
from .utils.db_connection import get_collection
from datetime import datetime, timedelta
import numpy as np

class Monitor(object):
    def __init__(self):
        self.games_collection = get_collection('games')
        self.rounds_collection = get_collection('rounds')

    def get_batch_results(self, batch_date):
        games = self.games_collection.find({'prediction_date':batch_date}, 
                                           {"_id":1, 'game_id':1, "status":1, 'bet_ratio':1, 'bookmaker_odds':1})                                
        num_all, num_completed, num_correct = 0, 0, 0
        revenue = 0
        for game in games:
            num_all += 1
            if game['status'] == 'undetermined':
                continue
            elif game['status'] == 'correct':
                num_correct += 1
                revenue += game['bet_ratio']*game['bookmaker_odds']
            num_completed += 1

        if num_completed == 0:
            ror = None
            acc = None
        else:
            ror = (revenue - 1)/1 * 100
            acc = num_correct/num_completed

        return {'batch_prediction_date': batch_date, 'games_completed':num_completed, 
                'all_games': num_all, 'rate_of_return': ror, 'accuracy': acc}
    
    def _get_time_interval(self, start_date, end_date):
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        delta = end_date_obj - start_date_obj
        num_days = delta.days + 1
        
        date_list = [start_date_obj + timedelta(days=x) for x in range(num_days)]
        return [datetime.strftime(date, '%Y-%#m-%#d') for date in date_list] 

    def get_performance_on_time_interval(self, start_date, end_date):
        batch_dates = self._get_time_interval(start_date, end_date)
    
        results = [self.get_batch_results(batch_date) for batch_date in batch_dates]

        # results = pd.DataFrame([self.get_batch_results(batch_date) for batch_date in batch_dates])
        # results = results[results['all_games'] > 0]
        
        num_completed, num_all = 0, 0
        profit = {'without_rule':0,
               'rules':{i:0 for i in range(10)}}
        ror = {'without_rule':0,
               'rules':{i:0 for i in range(10)}}
        num_correct = 0
        num_batches = 0
        batch_dates_played = []
        for result in results:
            if result['all_games'] == 0:
                continue
            num_all += result['all_games']
            if result['games_completed'] == 0:
                continue
            num_completed += result['games_completed']

            num_correct += np.round(result['games_completed'] * result['accuracy'], 1)

            ror['without_rule'] += result['rate_of_return']
            round_info = self.rounds_collection.find_one({'prediction_date':result['batch_prediction_date']}, 
                                                         {"_id":1, 'rule_in': 1, "values": 1, 'prediction_date':1})
            rule_in = round_info['rule_in']
            for k, v in dict(rule_in).items():
                profit['rules'][int(k)] += int(v)*result['rate_of_return']
                ror['rules'][int(k)] += int(v)
            
            num_batches += 1
            batch_dates_played.append(result['batch_prediction_date'])

        if num_completed == 0:
            ror = None
            acc = None
        else:
            acc = num_correct/num_completed
            for i in range(10):
                ror['rules'][i] = profit['rules'][i]/ror['rules'][i]

        return {'interval_start_date': start_date, 'interval_end_date': end_date,
                'games_completed':num_completed, 'all_games': num_all, 
                'profit':profit, 'rate_of_return': ror, 'accuracy': acc, 
                'num_bathces':num_batches, 'batch_dates': batch_dates_played}
        
