import pandas as pd
from matplotlib import pyplot as plt
from Fitness_function import *
from genetic_algorithm import run_ga


def different_combination(df_clean):
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
        print(f"Best Fitness = {res['best_fit']:.6f}")
        print(f"Best Chromosome = {[round(x, 4) for x in res['best_chrom']]}")
        print("-" * 40)

    return results


def sensitivity_analysis(df, chromosome, perturbation=0.05):
    base_fitness = fitness(df, chromosome)
    a_raw = chromosome[:6]
    b_raw = chromosome[6:]
    a_norm = normalize_coefficients(a_raw)
    b_norm = normalize_coefficients(b_raw)
    base_f1 = np.mean(calculate_f1_score(df, a_norm))
    base_f2 = np.mean(calculate_f2_score(df, b_norm))

    results_list = []
    for i in range(11):
        perturbed = chromosome.copy()
        perturbed[i] *= (1 + perturbation)
        perturbed[i] = np.clip(perturbed[i], 0.01, 10.0)

        new_fitness = fitness(df, perturbed)

        a_new = perturbed[:6]
        b_new = perturbed[6:]
        a_norm_new = normalize_coefficients(a_new)
        b_norm_new = normalize_coefficients(b_new)
        new_f1 = np.mean(calculate_f1_score(df, a_norm_new))
        new_f2 = np.mean(calculate_f2_score(df, b_norm_new))

        delta_fitness = new_fitness - base_fitness
        delta_f1 = new_f1 - base_f1
        delta_f2 = new_f2 - base_f2

        name = f'a{i + 1}' if i < 6 else f'b{i - 5}'
        group = 'f1' if i < 6 else 'f2'

        results_list.append({
            'Coefficient': f'{name} ({group})',
            'Original': chromosome[i],
            'Perturbed': perturbed[i],
            'ΔFitness': delta_fitness,
            'Δf1_score': delta_f1,
            'Δf2_score': delta_f2,
            'ΔFitness%': (delta_fitness / base_fitness) * 100
        })

    return pd.DataFrame(results_list), base_fitness, base_f1, base_f2


def coefficients_sensitive_analysis(df_clean , results):
    best_result = max(results, key=lambda x: x["best_fit"])
    print("\n" + "=" * 60)
    print(f"Best Result: {best_result['name']} Fitness = {best_result['best_fit']:.6f}")
    print("=" * 60)

    best_chromosome = best_result["best_chrom"]

    perturb = 0.05
    df_sensitivity, base_fit, base_f1, base_f2 = sensitivity_analysis(df_clean, best_chromosome, perturbation=perturb)

    print("\n--- Sensitivity Analysis (5% increase for each coefficient) ---".format(perturb))
    print(f"base values: fittness = {base_fit:.6f}, f1_score = {base_f1:.6f}, f2_score = {base_f2:.6f}\n")
    print(df_sensitivity.to_string(index=False, float_format="%.6f"))

    plt.figure(figsize=(12, 6))
    plt.bar(df_sensitivity['Coefficient'], df_sensitivity['ΔFitness'], color='steelblue')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.xlabel('coefficient')
    plt.ylabel('change in fittness')
    plt.title(f'Sensitivity Analysis: Effect of {perturb:.0%} increase of each coefficient on Fitness')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    x = np.arange(len(df_sensitivity['Coefficient']))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, df_sensitivity['Δf1_score'], width, label='Δf1_score', color='green')
    ax.bar(x + width / 2, df_sensitivity['Δf2_score'], width, label='Δf2_score', color='orange')
    ax.set_xlabel('coefficient')
    ax.set_ylabel('change in fittness')
    ax.set_title(f'Impact of {perturb:.0%} increase of each coefficient on f1_score and f2_score')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sensitivity['Coefficient'], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return best_chromosome


def critical_points_analysis(df_clean , best_chromosome):
    print("\n" + "=" * 80)
    print("Model behavior analysis on critical points of the dataset")
    print("=" * 80)

    # Using the best chromosome
    a_raw = best_chromosome[:6]
    b_raw = best_chromosome[6:]
    a_norm = normalize_coefficients(a_raw)
    b_norm = normalize_coefficients(b_raw)

    f1_raw_all = []
    f2_raw_all = []

    for idx, row in df_clean.iterrows():
        # Compute f1_raw components
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
        f1_raw_all.append(f1_raw)

        # Compute f2_raw
        f2_raw = (b_norm[0] * row['S_T'] +
                  b_norm[1] * row['S_H'] +
                  b_norm[2] * row['S_L'] +
                  b_norm[3] * row['S_C'] -
                  b_norm[4] * row['P_N'])
        f2_raw_all.append(f2_raw)

    # Compute ranges
    F1_min = np.min(f1_raw_all)
    F1_max = np.max(f1_raw_all)
    F2_min = np.min(f2_raw_all)
    F2_max = np.max(f2_raw_all)

    print(f"Range of f1_raw: [{F1_min:.3f}, {F1_max:.3f}]")
    print(f"Range of f2_raw: [{F2_min:.3f}, {F2_max:.3f}]")

    return F1_min, F1_max , F2_min, F2_max, a_norm , b_norm


