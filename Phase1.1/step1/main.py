import matplotlib.pyplot as plt
from Preprocessing import get_preprocessed_data
from genetic_algorithm import run_ga

DATA_PATH = 'data/1_26336110128.csv'

df_clean = get_preprocessed_data(DATA_PATH)

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

results = []


for cfg in configs:
    best_chrom, best_fit, best_hist, avg_hist = run_ga(
        df_clean,
        selection_method=cfg["selection"],
        crossover_method=cfg["crossover"],
        mutation_rate=cfg["mut"],
        generations=100,
        pop_size=50,
        crossover_rate=0.9,
        tournament_size=3,
        plot_convergence=False
    )
    results.append({
        "name": cfg["name"],
        "best_chrom": best_chrom,
        "best_fit": best_fit,
        "best_hist": best_hist,
        "avg_hist": avg_hist
    })

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, res in enumerate(results):
    ax = axes[i]
    generations = range(1, len(res["best_hist"]) + 1)
    ax.plot(generations, res["best_hist"], label='Best Fitness', color='blue', linewidth=2)
    ax.plot(generations, res["avg_hist"], label='Average Fitness', color='red', linestyle='--', linewidth=2)
    ax.set_title(res["name"])
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate(f'Best={res["best_fit"]:.4f}', xy=(0.7, 0.1), xycoords='axes fraction',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()


for res in results:
    print(f"{res['name']}:")
    print(f"   بهترین Fitness = {res['best_fit']:.6f}")
    print(f"   بهترین کروموزوم = {[round(x,4) for x in res['best_chrom']]}")
    print("-"*40)