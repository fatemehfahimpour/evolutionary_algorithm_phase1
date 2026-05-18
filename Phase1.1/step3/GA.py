import numpy as np
import random


class GA:
    def __init__(self, fitness_obj, pop_size=50, generations=100,
                 crossover_rate=0.9, mutation_rate=0.05,
                 selection_method='tournament', crossover_method="arithmetic", tournament_size=3):
        self.fitness_obj = fitness_obj
        self.crossover_method = crossover_method
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size

        self.bounds = fitness_obj.bounds
        self.dim = 4  # [T_in, H_in, L, CO2]

    def initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            individual = [
                random.uniform(*self.bounds['T_in']),
                random.uniform(*self.bounds['H_in']),
                random.uniform(*self.bounds['L']),
                random.uniform(*self.bounds['CO2'])
            ]
            pop.append(individual)
        return pop

    def evaluate_population(self, population):
        results = [self.fitness_obj.calculate_cost(ind) for ind in population]
        costs = [r[0] for r in results]
        f1_norms = [r[1] for r in results]
        f2_norms = [r[2] for r in results]
        return costs, f1_norms, f2_norms

    def tournament(self, population, costs):
        selected_indices = np.random.choice(
            len(population),
            size=self.tournament_size,
            replace=False
        )

        tournament_fitness = [
            costs[i]
            for i in selected_indices
        ]
        # finding best person min
        winner_index = selected_indices[np.argmin(tournament_fitness)]
        return population[winner_index]

    def roulette(self, population, costs):
        min_cost = min(costs)
        max_cost = max(costs)
        if max_cost - min_cost < 1e-8:
            return random.choice(population).copy()
        # بهترین فرد هزیته کمتری دارد پس max_cost - c برایش بیشتر است و احتمال انتخاب بیشتری دارد.
        fitness_vals = np.array([max_cost - c + 1e-8 for c in costs])
        probs = fitness_vals / np.sum(fitness_vals)
        idx = np.random.choice(len(population), p=probs)
        return population[idx].copy()

    def arithmetic(self, p1, p2):
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        alpha = random.random()
        c1 = [alpha * p1[i] + (1 - alpha) * p2[i] for i in range(self.dim)]
        c2 = [alpha * p2[i] + (1 - alpha) * p1[i] for i in range(self.dim)]
        return c1, c2

    def uniform(self, p1, p2):
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        child1 = []
        child2 = []
        for i in range(self.dim):
            if random.random() < 0.5:
                child1.append(p1[i])
                child2.append(p2[i])
            else:
                child1.append(p2[i])
                child2.append(p1[i])
        return child1, child2

    def gaussian_mutation(self, individual):
        mutated = individual.copy()
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                if i == 0:  # T_in
                    low, high = self.bounds['T_in']
                elif i == 1:  # H_in
                    low, high = self.bounds['H_in']
                elif i == 2:  # L
                    low, high = self.bounds['L']
                else:  # CO2
                    low, high = self.bounds['CO2']
                delta = np.random.normal(0, (high - low) * 0.05)
                mutated[i] += delta
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated

    def run(self):
        population = self.initialize_population()
        costs, f1_norms, f2_norms = self.evaluate_population(population)

        best_cost_history = []
        avg_cost_history = []
        best_f1_norm_history = []
        best_f2_norm_history = []

        for gen in range(self.generations):
            best_idx = np.argmin(costs)
            best_cost_history.append(costs[best_idx])
            avg_cost_history.append(np.mean(costs))

            # ذخیره f1_norm و f2_norm بهترین فرد این نسل
            best_f1_norm_history.append(f1_norms[best_idx])
            best_f2_norm_history.append(f2_norms[best_idx])

            new_population = []

            while len(new_population) < self.pop_size:
                if self.selection_method == 'tournament':
                    parent1 = self.tournament(population, costs)
                    parent2 = self.tournament(population, costs)
                else:
                    parent1 = self.roulette(population, costs)
                    parent2 = self.roulette(population, costs)

                if self.crossover_method == "arithmetic":
                    child1, child2 = self.arithmetic(parent1, parent2)
                else:
                    child1, child2 = self.uniform(parent1, parent2)

                child1 = self.gaussian_mutation(child1)
                child2 = self.gaussian_mutation(child2)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            population = new_population
            costs, f1_norms, f2_norms = self.evaluate_population(population)

        best_idx = np.argmin(costs)
        best_individual = population[best_idx]
        best_cost = costs[best_idx]

        return best_individual, best_cost, best_cost_history, avg_cost_history, best_f1_norm_history, best_f2_norm_history

    def run_multiple(self, num_runs=5):  # میانگین 5بار اجرا
        best_individuals = []
        best_costs = []
        all_histories = []
        for run in range(num_runs):
            ind, cost, hist, _ = self.run()
            best_individuals.append(ind)
            best_costs.append(cost)
            all_histories.append(hist)
        mean_cost = np.mean(best_costs)
        std_cost = np.std(best_costs)

        global_best_idx = np.argmin(best_costs)
        global_best_individual = best_individuals[global_best_idx]
        global_best_cost = best_costs[global_best_idx]
        return {
            'best_individual': global_best_individual,
            'best_cost': global_best_cost,
            'mean_cost': mean_cost,
            'std_cost': std_cost,
            'all_costs': best_costs,
            'all_individuals': best_individuals,
            'histories': all_histories
        }
