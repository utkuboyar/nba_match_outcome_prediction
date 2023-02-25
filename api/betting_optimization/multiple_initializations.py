import numpy as np
import multiprocessing
from .genetic_optimizer import Process

class BettingOptimizer(object):
    def __init__(self, probs, odds, num_inits=10, seed=0):
        self.num_inits = num_inits
        self.seed = seed

        probs, odds = np.array(probs), np.array(odds)

        self.n = probs.shape[0]
        self.A = probs*odds-1
        self.B = (1-probs)*probs*np.square(odds)

        self.idx = self.A > 0

        self.ga_config = {'A':self.A[self.idx], 
                          'B':self.B[self.idx],
                          'population_size':100,
                          'max_iter':10000,
                          'tolerance':10,
                          'seed':10}

    def run(self):
        if self.seed:
            np.random.seed(self.seed)
        self.ga_seeds = np.random.randint(0, self.num_inits*100000, self.num_inits)

        args_list = []
        for seed in self.ga_seeds:
            config = self.ga_config.copy()
            config['seed'] = seed
            args_list.append(config)

        with multiprocessing.Pool() as pool:
            results = pool.map(Process.process_one, args_list)

        for res in results:
            if res[1] == -10000:
                return 'all expected profits are negative'

        ratios_returned, value = max(results, key = lambda i : i[1])
        ratios = np.zeros(self.n)
        ratios[self.idx] = ratios_returned
        return (ratios, value)
