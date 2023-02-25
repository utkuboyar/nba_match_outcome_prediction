import numpy as np
from itertools import combinations
from time import time
from sys import maxsize

class GeneticAlgorithm(object):
        
    def _fitness(self, x):
        return np.dot(self.A,x)/np.sqrt(np.dot(self.B,np.square(x)))
    
    def _crossover(self, x1, x2):
        x_new = (x1 + x2)/2
        return x_new
        
    def _mutation(self, x):
        total_mutation = 0.2
        
        num_positive_mutation = np.random.randint(1, self.n)
        num_negative_mutation = self.n - num_positive_mutation
        
        positive_mutation_cells = np.random.choice(self.n, num_positive_mutation, replace=False)
        
        positive_mutations = np.random.random(num_positive_mutation)
        positive_mutations *= total_mutation/2 * 1/np.sum(positive_mutations)
        
        negative_mutations = np.random.random(num_negative_mutation)
        negative_mutations *= total_mutation/2 * 1/np.sum(negative_mutations)
        negative_mutations = -negative_mutations
        
        all_mutations = []
        pos_counter, neg_counter = 0, 0
        for i in range(self.n):
            if i in positive_mutation_cells:
                all_mutations.append(positive_mutations[pos_counter])
                pos_counter += 1
            else:
                all_mutations.append(negative_mutations[neg_counter])
                neg_counter += 1
        
        x_mutated = x + np.array(all_mutations)
        
        return x_mutated
    
    def _recover(self, x_mutated):
        infeasible = np.where(x_mutated < 0)
        while len(infeasible[0]) > 0:
            rest = np.where(x_mutated > 0)
            infeasible_negative_amount = np.sum(x_mutated[infeasible])
            x_mutated[infeasible] = np.zeros(len(infeasible))
            
            remaining = x_mutated[rest]

            num_recovery_mutation = np.random.randint(1, len(remaining)+1)
            recovery_mutation_cells = np.random.choice(range(len(remaining)), num_recovery_mutation, replace=False)
            recovery_mutation = np.random.random(num_recovery_mutation)
            recovery_mutation = recovery_mutation/np.sum(recovery_mutation) * infeasible_negative_amount

            remaining[recovery_mutation_cells] += recovery_mutation
            x_mutated[rest] = remaining

            infeasible = np.where(x_mutated < 0)
    
        return x_mutated
    
    
    def _breed(self, x1, x2):
        x_new = self._crossover(x1, x2)
        x_new = self._mutation(x_new)
        x_new = self._recover(x_new)
        return x_new
    
    def _initialize(self):
        self.population = []
        self.sample_fitness = []
        
        for i in range(self.population_size):
            sample = np.random.random(self.n)
            sample /= np.sum(sample)
            self.population.append(sample)
            self.sample_fitness.append(self._fitness(sample))
            
        self.population = np.array(self.population)
        self.sample_fitness = np.array(self.sample_fitness)
        
    def _binary_search(self, cdf, val):
        lowest, highest = 0, len(cdf)-1
        while True:
            i = (highest+lowest)//2
            if val > cdf[i]:
                if val <= cdf[i+1]:
                    return i + 1
                else:
                    lowest = i
            else:
                if i == 0:
                    return 0
                highest = i
        
        
    def _fitness_proportional_selection(self):
        cdf = self.sample_fitness - np.min(self.sample_fitness)
        cdf /= np.sum(cdf)
        cdf = np.cumsum(cdf)

        pairs = []
        for _ in range(self.population_size):
            rand1 = np.random.random()
            parent1 = self._binary_search(cdf, rand1)

            rand2 = np.random.random()
            parent2 = self._binary_search(cdf, rand2)
            while parent2 == parent1:
                rand2 = np.random.random()
                parent2 = self._binary_search(cdf, rand2)

            pair = [parent1, parent2]
            pairs.append(pair)

        return pairs
        
    def _create_new_generation(self):
        pairs = self._fitness_proportional_selection()
        
        for pair in pairs:
            parent1, parent2 = self.population[pair]
            x_new = self._breed(parent1, parent2)
            
            self.population = np.append(self.population, x_new.reshape(1,-1), axis=0)
            x_new_fitness = self._fitness(x_new)
            self.sample_fitness = np.append(self.sample_fitness, x_new_fitness)
            
    def _natural_selection(self):
        self._prev_best = self.sample_fitness[0].round(5)
        
        idx = np.argsort(-self.sample_fitness)[:self.population_size]
        
        self.sample_fitness = self.sample_fitness[idx]
        self.population = self.population[idx]
        
    def _converged(self):
        if self.sample_fitness[0].round(5) <= self._prev_best:
            self._non_improvement_counter += 1
        else:
            self._non_improvement_counter = 0
        
        return self._non_improvement_counter == self._tolerance
    
    def _make_sense(self):
        if np.sum(self.A > 0) == 0:
            print('!!!!!!!')
            print(self.A)
            print(self.B)
            # raise Exception('expected profits for games are all negative')
            return 'all negative'
             
        if self.n == 1:
            self.population = np.ones((1,1))
            self.sample_fitness = self._fitness(np.ones((1,1)))
            return 'terminate'
        
        return 'ok'
            
    def run(self, population_size, max_iter, tolerance):
        sensible = self._make_sense()
        if sensible == 'terminate':
            x = np.ones(1)
            return (x, self._fitness(x=x))
        if sensible == 'all negative':
            self.population, self.sample_fitness = [None], [-10000]
            return 'all negative'
        
        self.population_size = population_size
        self._tolerance = tolerance
        self._non_improvement_counter = 0
        
        self._initialize()
        for _ in range(max_iter):
            self._create_new_generation()
            self._natural_selection()
            if self._converged():
                break

    def __call__(self, A, B, population_size=100, max_iter=10000, tolerance=10):
        self.A = A
        self.B = B
        self.n = A.shape[0]
        self.run(population_size=population_size, max_iter=max_iter, tolerance=tolerance)
        
        return self.population[0], self.sample_fitness[0]
        
        
class Process:
    @staticmethod
    def process_one(ga_config):
        ga = GeneticAlgorithm()

        A = ga_config['A']
        B = ga_config['B']
        population_size = ga_config['population_size']
        max_iter = ga_config['max_iter']
        tolerance = ga_config['tolerance']

        seed = ga_config['seed']

        np.random.seed(seed)
        return ga(A=A, B=B, population_size=population_size,
                              max_iter=max_iter, tolerance=tolerance)