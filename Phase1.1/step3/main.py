import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from step1.Preprocessing import get_preprocessed_data
from step1.Fitness_function import normalize_coefficients
from step1.genetic_algorithm import run_ga
from step3.fitness import Fitness
from GA import GA
from scenarios import get_scenarios_for_team2

# ---------------بدست آوردن ضرایب-----------------
DATA_PATH_STEP1 = Path(__file__).resolve().parent.parent / 'step1' / 'data' / '1_26336110128.csv'
df_clean = get_preprocessed_data(str(DATA_PATH_STEP1))

best_chrom, best_fit, best_hist, avg_hist = run_ga(
    df_clean,
    pop_size=80,
    generations=150,
    selection_method="tournament",
    crossover_method="arithmetic",
    crossover_rate=0.9,
    mutation_rate=0.05,
    tournament_size=3,
    plot_convergence=False
)
print(f"chromosome:{[round(x, 4) for x in best_chrom]}")
print(f"fitness:{best_fit:.6f}\n")

ai = best_chrom[:6]
bj = best_chrom[6:]
a_norm = normalize_coefficients(ai)
b_norm = normalize_coefficients(bj)

# -----------------بازه ی f1 , f2------------------
f1 = []
f2 = []

for idx, row in df_clean.iterrows():
    p_e = max(0, row['E'] - 100)
    p_delta_t = abs(row['T_in'] - row['T_out'])
    p_delta_h = abs(row['H_in'] - row['H_out'])
    p_light_def = max(0, 500 - (row['L'] + row['Solar']))
    p_n = np.sqrt(row['N'])
    p_vent_risk = max(0, row['N'] * row['Wind'] - 50)
    f1_raw = (a_norm[0] * np.log1p(p_e ** 2) +
              a_norm[1] * p_delta_t +
              a_norm[2] * p_delta_h +
              a_norm[3] * p_light_def +
              a_norm[4] * p_n +
              a_norm[5] * p_vent_risk)
    f1.append(f1_raw)

    f2_raw = (b_norm[0] * row['S_T'] +
              b_norm[1] * row['S_H'] +
              b_norm[2] * row['S_L'] +
              b_norm[3] * row['S_C'] -
              b_norm[4] * row['P_N'])
    f2.append(f2_raw)

F1_MIN = np.min(f1)
F1_MAX = np.max(f1)
F2_MIN = np.min(f2)
F2_MAX = np.max(f2)

# ----------------اجرای مرحله سوم روی 4 سناریو----------------
# all_scenarios = get_scenarios_for_team2()
# scenarios = all_scenarios()
scenarios = get_scenarios_for_team2()

configs = [
    {"name": "A: Tournament + Arithmetic + mut=0.05",
     "selection": "tournament", "crossover": "arithmetic", "mut": 0.05},
    {"name": "B: Tournament + Uniform + mut=0.01",
     "selection": "tournament", "crossover": "uniform", "mut": 0.01},
    {"name": "C: Roulette + Arithmetic + mut=0.05",
     "selection": "roulette", "crossover": "arithmetic", "mut": 0.05},
    {"name": "D: Roulette + Uniform + mut=0.01",
     "selection": "roulette", "crossover": "uniform", "mut": 0.01},
]

for sc in scenarios:
    print(f"\n{sc['name']}")
    fitness_obj = Fitness(best_chrom, sc['z'], sc['bounds'], F1_MIN, F1_MAX, F2_MIN, F2_MAX)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    sc['name'] = sc['name'][::-1]

    for idx, cfg in enumerate(configs):
        print(f"{cfg['name']}")
        ga = GA(fitness_obj,
                pop_size=100,
                generations=200,
                crossover_rate=0.9,
                mutation_rate=cfg['mut'],
                selection_method=cfg['selection'],
                crossover_method=cfg['crossover'],
                tournament_size=3)
        best_ind, best_cost, best_hist, avg_hist = ga.run()

        ax = axes[idx]
        ax.plot(best_hist, label='Best Cost', color='blue', linewidth=2)
        ax.plot(avg_hist, label='Avg Cost', color='red', linestyle='--', linewidth=2)
        title = cfg['name']
        ax.set_title(f"{title}\nBest Cost = {best_cost:.4f}")
        ax.set_xlabel('Generation')
        ax.set_ylabel('Cost')
        ax.legend()
        ax.grid(True, alpha=0.3)
        print(f"    بهترین هزینه: {best_cost:.6f}")

    plt.suptitle(f"{sc['name']}", fontsize=14)
    plt.tight_layout()
    plt.show()
