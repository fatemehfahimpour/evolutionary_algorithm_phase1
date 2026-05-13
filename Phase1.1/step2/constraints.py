import random
import numpy as np
from pathlib import Path
from step1.Preprocessing import get_preprocessed_data
from step1.genetic_algorithm import run_ga

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'step1' / 'data' / '1_26336110128.csv'

# df_clean = get_preprocessed_data(DATA_PATH)
# # اینجا باید بهترین ورودی هایی که از تحلیل بخش 1 بدست آمده استفاده بشه و مقادیرش به این متد داده بشه
# population, fitness_values, best_fitness_history, avg_fitness_history = run_ga(df_clean, plot_convergence=False)
# a_coefficients = population[:6]
# b_coefficients = population[6:]


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


population = generate_population(1000)
