import random
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from step1.Preprocessing import calculate_energy_column
from step1.Preprocessing import get_preprocessed_data
from step1.genetic_algorithm import run_ga
from step1.genetic_algorithm import tournament_selection
from step1.genetic_algorithm import uniform_crossover
from step1.Fitness_function import calculate_f2_score
from step1.genetic_algorithm import arithmetic_crossover
from step1.genetic_algorithm import roulette_wheel_selection

from step1.Preprocessing import get_k_w

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'step1' / 'data' / '1_26336110128.csv'

df_clean = get_preprocessed_data(DATA_PATH)
# # اینجا باید بهترین ورودی هایی که از تحلیل بخش 1 بدست آمده استفاده بشه و مقادیرش به این متد داده بشه
population, fitness_values, best_fitness_history, avg_fitness_history = run_ga(df_clean, plot_convergence=False)
a_coefficients = population[:6]
b_coefficients = population[6:]


# chromosome format: [T_in, T_out, H_in, L, N, CO2, W_night, W_sunny, W_cloudy, W_humid, W_rainy, W_stormy, cold]
def generate_individual():
    T_in = random.uniform(18, 30)
    T_out = random.uniform(0, 40)
    H_in = random.uniform(20, 80)
    L = random.uniform(100, 900)
    N = random.randint(1, 30)
    CO2 = random.uniform(400, 1500)

    weather_options = ["night", "sunny", "cloudy", "humid", "rainy", "stormy", "cold"]
    weather_vector = [0] * len(weather_options)
    selected_weather = random.randint(0, len(weather_options) - 1)
    weather_vector[selected_weather] = 1

    chromosome = [T_in, T_out, H_in, L, N, CO2] + weather_vector
    return chromosome


def generate_population(pop_size):
    population = []

    for _ in range(pop_size):
        individual = generate_individual()

        population.append(individual)

    return np.array(population)

def initialize_population(pop_size):
    return [generate_individual() for _ in range(pop_size)]


def fitness_stage2():
    return


def inversion_mutation(chromosome, mutation_rate=0.05, low=0, high=10):
    mutated = np.array(chromosome, dtype=float)
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[i] = high - mutated[i]
    return mutated.tolist()


def run_ga_stage2(pop_size=80, generations=150,
                  selection_method='tournament', crossover_method='arithmetic',
                  crossover_rate=0.9, mutation_rate=0.05, tournament_size=3,
                  lambda_real=0.0, real_penalty=None, plot_convergence=True):

    population = initialize_population(pop_size)
    fitness_vals = [fitness_stage2(ind, lambda_real, real_penalty) for ind in population]
    best_fitness_history = []
    avg_fitness_history = []
    for gen in range(generations):
        best_fitness_history.append(max(fitness_vals))
        avg_fitness_history.append(np.mean(fitness_vals))
        new_pop = []
        new_fit = []
        while len(new_pop) < pop_size:
            if selection_method == 'tournament':
                p1 = tournament_selection(population, fitness_vals, tournament_size)
                p2 = tournament_selection(population, fitness_vals, tournament_size)
            else:
                p1 = roulette_wheel_selection(population, fitness_vals)
                p2 = roulette_wheel_selection(population, fitness_vals)
            if random.random() < crossover_rate:
                if crossover_method == 'arithmetic':
                    c1, c2 = arithmetic_crossover(p1, p2)
                else:
                    c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = inversion_mutation(c1, mutation_rate)
            c2 = inversion_mutation(c2, mutation_rate)

            fit1 = fitness_stage2(c1, lambda_real, real_penalty)
            fit2 = fitness_stage2(c2, lambda_real, real_penalty)
            new_pop.append(c1)
            new_fit.append(fit1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
                new_fit.append(fit2)
        population = new_pop
        fitness_vals = new_fit
    best_idx = np.argmax(fitness_vals)
    best_individual = population[best_idx]
    best_fitness = fitness_vals[best_idx]
    if plot_convergence:
        plt.figure(figsize=(12,5))
        plt.plot(best_fitness_history, label='Best Fitness')
        plt.plot(avg_fitness_history, label='Avg Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Convergence - Stage2 GA')
        plt.legend()
        plt.grid()
        plt.show()
    return best_individual, best_fitness, best_fitness_history, avg_fitness_history


# population = generate_population(1000)
