import numpy as np
from scipy.stats import t

from .multiple_initializations import BettingOptimizer

class Portfolio(object):
    def __init__(self, probs, odds):
        self.rules={'headers': ['min_n_played', 'min_lower_0.9', 'logical_operator'],
                    'values':{'0': [7, -0.6, 'or'],
                            '1': [2, -0.6, 'and'],
                            '2': [1, -0.6, 'and'],
                            '3': [7, -0.5, 'or'],
                            '4': [1, -0.5, 'and'],
                            '5': [2, -0.5, 'and'],
                            '6': [4, -0.3, 'or'],
                            '7': [4, 0.0, 'or'],
                            '8': [4, -0.2, 'or'],
                            '9': [4, -0.1, 'or']}}
        probs, odds = np.array(probs), np.array(odds)
        self.probs = probs
        self.odds = odds
        self.n = probs.shape[0]
        
    def _get_lower_lim(self, ratios):
        e = np.sum((self.probs * self.odds - 1) * ratios)
        std = np.sqrt(np.sum((np.ones(self.n) - self.probs) * self.probs * np.square(ratios) * np.square(self.odds)))

        conf_level = 0.9
        if self.n > 1:
            dof = self.n - 1
        else:
            dof = 1
        t_value = t.ppf((1+conf_level)/2, dof)
        
        lower_lim = e - t_value * std * np.sqrt(1 + 1/dof)
        lower_lim = max(lower_lim, -1)
        return lower_lim

    def _pass_to_rules(self, n_played, lower_lim):
        self.rule_in = {}
        for k, v in self.rules['values'].items():
            min_n_played, min_lower_lim, logical_op = v
            c1 = n_played >= min_n_played
            c2 = lower_lim > min_lower_lim
            if logical_op == 'and':
                self.rule_in[k] = str(int(c1 and c2))
            else:
                self.rule_in[k] = str(int(c1 or c2))
        
    def run(self, game_ids):
        optimizer = BettingOptimizer(probs=self.probs, odds=self.odds)
        result = optimizer.run()

        if result == 'all expected profits are negative':
            ratios_to_return = {game_id: (0, bookmaker_odds) for game_id, bookmaker_odds in zip(game_ids, self.odds)}
        else:
            ratios, _ = result
            ratios_to_return = {game_id: (ratios[i], self.odds[i]) for game_id, i in zip(game_ids, range(len(game_ids)))}
        
        n_played = np.sum(ratios > 0)
        lower_lim = self._get_lower_lim(ratios)
        
        self._pass_to_rules(n_played, lower_lim)
        decisions = {'rules':self.rule_in, 'n_played':n_played, 'lower_lim':lower_lim}
        return ratios_to_return, decisions


        