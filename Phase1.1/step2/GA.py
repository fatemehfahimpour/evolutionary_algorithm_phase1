import random
import numpy as np
from pathlib import Path
from step1.Preprocessing import get_preprocessed_data
from step1.genetic_algorithm import (
    tournament_selection,
    uniform_crossover,
    arithmetic_crossover,
    roulette_wheel_selection
)
from step2.fitness_function import *

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'step1' / 'data' / '1_26336110128.csv'
df_clean = get_preprocessed_data(DATA_PATH)
# chromosome = [T_in, T_out, H_in, L, N, CO2, wind, solar, H_out] + weather_vector
# N
WEATHER_LIST = ["night", "sunny", "cloudy", "humid", "rainy", "stormy", "cold"]


def initialize_population():
    T_in = random.uniform(10, 35)
    T_out = random.uniform(-5, 45)
    H_in = random.uniform(10, 90)
    L = random.uniform(0, 1000)
    N = random.randint(1, 30)
    CO2 = random.uniform(300, 1700)
    wind = random.uniform(df_clean['Wind'].min(), df_clean['Wind'].max())
    solar = random.uniform(df_clean['Solar'].min(), df_clean['Solar'].max())
    H_out = random.uniform(df_clean['H_out'].min(), df_clean['H_out'].max())

    weather_vector = [0] * len(WEATHER_LIST)
    selected_weather = random.randint(0, len(WEATHER_LIST) - 1)
    weather_vector[selected_weather] = 1

    chromosome = [T_in, T_out, H_in, L, N, CO2, wind, solar, H_out] + weather_vector
    return chromosome


def evaluate_population(pop_size):
    return [initialize_population() for _ in range(pop_size)]


def gaussian_mutation(individual, mutation_rate=0.05):
    mutated = individual.copy()
    for i in range(9):
        if random.random() < mutation_rate:
            if i == 0:
                low, high = 10, 35  # T_in
            elif i == 1:
                low, high = -5, 45  # T_out
            elif i == 2:
                low, high = 10, 90  # H_in
            elif i == 3:
                low, high = 0, 1000  # L
            elif i == 4:
                low, high = 1, 30  # N
            elif i == 5:
                low, high = 300, 1700  # CO2
            elif i == 6:
                low, high = df_clean['Wind'].min(), df_clean['Wind'].max()  # Wind
            elif i == 7:
                low, high = df_clean['Solar'].min(), df_clean['Solar'].max()  # Solar
            else:
                low, high = df_clean['H_out'].min(), df_clean['H_out'].max()  # H_out
            delta = np.random.normal(0, (high - low) * 0.05)
            mutated[i] += delta
            if i == 4:  # N صحیح
                mutated[i] = int(round(np.clip(mutated[i], low, high)))
            else:
                mutated[i] = np.clip(mutated[i], low, high)

    if random.random() < mutation_rate:
        new_idx = random.randint(0, len(WEATHER_LIST) - 1)
        mutated[9:] = [0] * len(WEATHER_LIST)
        mutated[9 + new_idx] = 1
    return mutated


def compute_objective_ranges(population, a, b):
    f1_vals = [calculate_f1(ch, a) for ch in population]
    f2_vals = [calculate_f2(ch, b) for ch in population]

    F1_min = min(f1_vals)
    F1_max = max(f1_vals)
    F2_min = min(f2_vals)
    F2_max = max(f2_vals)

    return F1_min, F1_max, F2_min, F2_max


def run_ga_step2(a, b, F1_min, F1_max, F2_min, F2_max,
                 pop_size=80, generations=150,
                 selection_method='tournament',
                 crossover_method='arithmetic',
                 crossover_rate=0.9,
                 mutation_rate=0.05,
                 tournament_size=3,
                 elitism_rate=0.05):

    delta = 1e-5

    population = evaluate_population(pop_size)
    fitness_vals = [
        fitness_function(ind, a, b, F1_min, F1_max, F2_min, F2_max)
        for ind in population
    ]

    best_fitness_history = []
    avg_fitness_history = []

    # history of best individual
    best_f1_history = []
    best_f2_history = []

    # history of population mean (this is what you called f_bar)
    mean_f1_history = []
    mean_f2_history = []

    # normalized mean histories
    norm_mean_f1_history = []
    norm_mean_f2_history = []

    for gen in range(generations):
        # محاسبه مقادیر خام کل جمعیت
        f1_vals = [calculate_f1(ind, a) for ind in population]
        f2_vals = [calculate_f2(ind, b) for ind in population]

        f1_bar = np.mean(f1_vals)
        f2_bar = np.mean(f2_vals)

        # نرمال‌سازی میانگین‌ها بر اساس فرمول صورت سوال
        f1_bar_norm = (f1_bar - F1_min) / (F1_max - F1_min + delta)
        f2_bar_norm = (f2_bar - F2_min) / (F2_max - F2_min + delta)

        # اگر خواستی حتما در بازه [0,1] بماند
        f1_bar_norm = np.clip(f1_bar_norm, 0, 1)
        f2_bar_norm = np.clip(f2_bar_norm, 0, 1)

        mean_f1_history.append(f1_bar)
        mean_f2_history.append(f2_bar)
        norm_mean_f1_history.append(f1_bar_norm)
        norm_mean_f2_history.append(f2_bar_norm)

        # -----------------------------
        # تاریخچه fitness
        # -----------------------------
        best_fitness_history.append(np.max(fitness_vals))
        avg_fitness_history.append(np.mean(fitness_vals))

        # بهترین فرد نسل
        best_idx = np.argmax(fitness_vals)
        best_ind = population[best_idx]

        best_f1 = calculate_f1(best_ind, a)
        best_f2 = calculate_f2(best_ind, b)

        best_f1_norm = (best_f1 - F1_min) / (F1_max - F1_min + delta)
        best_f2_norm = (best_f2 - F2_min) / (F2_max - F2_min + delta)

        best_f1_norm = np.clip(best_f1_norm, 0, 1)
        best_f2_norm = np.clip(best_f2_norm, 0, 1)

        best_f1_history.append(best_f1_norm)
        best_f2_history.append(best_f2_norm)

        # ELITISM
        elite_count = max(1, int(pop_size * elitism_rate))
        sorted_idx = np.argsort(fitness_vals)[::-1]

        elites = [population[i].copy() for i in sorted_idx[:elite_count]]
        elites_fit = [fitness_vals[i] for i in sorted_idx[:elite_count]]

        new_pop = elites.copy()
        new_fit = elites_fit.copy()

        # تولید بقیه جمعیت
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

            c1 = gaussian_mutation(c1, mutation_rate)
            c2 = gaussian_mutation(c2, mutation_rate)

            fit1 = fitness_function(c1, a, b, F1_min, F1_max, F2_min, F2_max)
            fit2 = fitness_function(c2, a, b, F1_min, F1_max, F2_min, F2_max)

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

    return (
        best_individual,
        best_fitness,
        best_fitness_history,
        avg_fitness_history,
        best_f1_history,
        best_f2_history,
    )

# def run_ga_step2(a, b , F1_min , F1_max, F2_min, F2_max, pop_size=80, generations=150,
#                  selection_method='tournament', crossover_method='arithmetic',
#                  crossover_rate=0.9, mutation_rate=0.05, tournament_size=3):
#
#     population = evaluate_population(pop_size)
#     fitness_vals = [
#         fitness_function(ind, a, b, F1_min, F1_max, F2_min, F2_max)
#         for ind in population
#     ]
#
#     best_fitness_history = []
#     avg_fitness_history = []
#     f1_history = []
#     f2_history = []
#
#     for gen in range(generations):
#         #history
#         best_fitness_history.append(max(fitness_vals))
#         avg_fitness_history.append(np.mean(fitness_vals))
#         best_idx = np.argmax(fitness_vals)
#         best_ind = population[best_idx]
#         f1 = calculate_f1(best_ind, a)
#         f2 = calculate_f2(best_ind, b)
#         delta = 1e-6
#         f1_history.append((f1 - F1_min) / (F1_max - F1_min + delta))
#         f2_history.append((f2 - F2_min) / (F2_max - F2_min + delta))
#
#         new_pop = []
#         new_fit = []
#
#         while len(new_pop) < pop_size:
#             if selection_method == 'tournament':
#                 p1 = tournament_selection(population, fitness_vals, tournament_size)
#                 p2 = tournament_selection(population, fitness_vals, tournament_size)
#             else:
#                 p1 = roulette_wheel_selection(population, fitness_vals)
#                 p2 = roulette_wheel_selection(population, fitness_vals)
#
#             if random.random() < crossover_rate:
#                 if crossover_method == 'arithmetic':
#                     c1, c2 = arithmetic_crossover(p1, p2)
#                 else:
#                     c1, c2 = uniform_crossover(p1, p2)
#             else:
#                 c1, c2 = p1.copy(), p2.copy()
#
#             c1 = gaussian_mutation(c1, mutation_rate)
#             c2 = gaussian_mutation(c2, mutation_rate)
#
#             fit1 = fitness_function(c1, a, b, F1_min, F1_max, F2_min, F2_max)
#             fit2 = fitness_function(c2, a, b, F1_min, F1_max, F2_min, F2_max)
#
#             new_pop.append(c1)
#             new_fit.append(fit1)
#             if len(new_pop) < pop_size:
#                 new_pop.append(c2)
#                 new_fit.append(fit2)
#
#         population = new_pop
#         fitness_vals = new_fit
#
#     best_idx = np.argmax(fitness_vals)
#     best_individual = population[best_idx]
#     best_fitness = fitness_vals[best_idx]
#
#     return best_individual, best_fitness, best_fitness_history, avg_fitness_history, f1_history, f2_history
