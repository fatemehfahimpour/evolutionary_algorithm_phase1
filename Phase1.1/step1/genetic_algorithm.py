import numpy as np
from Fitness_function import fitness


def roulette_wheel_selection(population, fitness_values):
    """
    population: chromosome list
    fitness_values: chromosome's fitness values
    output = chosen chromosome
    """
    # اینجا براساس برازندگی به هر عضو شانس انتخاب شدن داده میشه و به صورت تصادفی انتخاب صورت میگیره
    fitness_values = np.array(fitness_values, dtype=float)

    fitness_values = np.maximum(fitness_values, 0)

    total_fitness = np.sum(fitness_values)

    if total_fitness == 0:
        return population[np.random.randint(len(population))]

    probabilities = fitness_values / total_fitness

    selected_index = np.random.choice(
        len(population),
        p=probabilities
    )

    return population[selected_index]


def tournament_selection(population, fitness_values, tournament_size=3):
    # یک گروه 3 تایی انتخاب میشه هر بار و از بین این گروه اونی که بهترین فیتنس را داره انتخاب میشه
    selected_indices = np.random.choice(
        len(population),
        size=tournament_size,
        replace=False
    )

    tournament_fitness = [
        fitness_values[i]
        for i in selected_indices
    ]

    # finding best person
    winner_index = selected_indices[np.argmax(tournament_fitness)]

    return population[winner_index]


def uniform_crossover(parent1, parent2, crossover_rate=0.5):
    child1 = []
    child2 = []

    for gene1, gene2 in zip(parent1, parent2):

        if np.random.rand() < crossover_rate:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)

    return child1, child2


def arithmetic_crossover(parent1, parent2, alpha=None):
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    if alpha is None:
        alpha = np.random.rand()

    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1

    return child1.tolist(), child2.tolist()


def inversion_mutation(chromosome, mutation_rate=0.05, low=0, high=10):
    mutated = np.array(chromosome, dtype=float)

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[i] = high - mutated[i]

    # اعمال محدودیت بازه
    mutated = np.clip(mutated, low, high)

    return mutated.tolist()


def initialize_population(df_clean, pop_size=50, low=0, high=10):
    population = []
    fitness_values = []

    for _ in range(pop_size):
        chrom = np.random.uniform(low, high, size=11).tolist()
        fit = fitness(df_clean, chrom)
        population.append(chrom)
        fitness_values.append(fit)

    return population, fitness_values



