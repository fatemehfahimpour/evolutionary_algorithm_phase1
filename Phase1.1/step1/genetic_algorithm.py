import numpy as np


def roulette_wheel_selection(population, fitness_values):
    """
    population: chromosome list
    fitness_values: chromosome's fitness values
    output = chosen chromosome
    """
    #اینجا براساس برازندگی به هر عضو شانس انتخاب شدن داده میشه و به صورت تصادفی انتخاب صورت میگیره
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
    #یک گروه 3 تایی انتخاب میشه هر بار و از بین این گروه اونی که بهترین فیتنس را داره انتخاب میشه
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
