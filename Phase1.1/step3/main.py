import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
from step1.Preprocessing import get_preprocessed_data
from step1.Fitness_function import normalize_coefficients
from step1.genetic_algorithm import run_ga as run_ga_stage1
from step3.fitness import Fitness
from GA import GA
from scenarios import get_scenarios_for_team2

# ---------------بدست آوردن ضرایب از مرحله اول-----------------
DATA_PATH_STEP1 = Path(__file__).resolve().parent.parent / 'step1' / 'data' / '1_26336110128.csv'
df_clean = get_preprocessed_data(str(DATA_PATH_STEP1))

best_chrom, best_fit, best_hist, avg_hist, best_f1_history, best_f2_history = run_ga_stage1(
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

# ----------------توابع کمکی برای محاسبه بازه مرحله 3-----------------
def calculate_raw_f1_f2_for_population(fitness_obj, population):
    """محاسبه مقادیر خام f1 و f2 برای یک جمعیت"""
    f1_vals = []
    f2_vals = []

    # محاسبه دستی f1 و f2 بدون نرمال‌سازی
    for ind in population:
        T_in, H_in, L, CO2 = ind

        # محاسبه انرژی
        delta_T = abs(T_in - fitness_obj.z['T_out'])
        P_hum = max(0, H_in - 60) ** 2 + max(0, 30 - H_in) ** 2
        energy = fitness_obj.weather * (0.8 * delta_T ** 2 + 12 * np.log(1 + fitness_obj.z['N']) + 0.02 * P_hum)
        energy += 6 * np.log(1 + L) + 2 * np.log(1 + CO2)

        # محاسبه امتیازهای کیفیت
        # S_T
        if 20 <= T_in <= 24:
            ST = 1
        elif 18 <= T_in < 20 or 24 < T_in <= 26:
            ST = 0.6
        elif 16 <= T_in < 18 or 26 < T_in <= 28:
            ST = 0.2
        else:
            ST = 0

        # S_H
        if 45 <= H_in <= 60:
            SH = 1
        elif 35 <= H_in < 45 or 60 < H_in <= 70:
            SH = 0.5
        else:
            SH = 0

        # S_C
        if 800 <= CO2 <= 1200:
            SC = 1
        elif 700 <= CO2 < 800 or 1200 < CO2 <= 1400:
            SC = 0.4
        else:
            SC = 0

        # S_L
        total_light = L + fitness_obj.z['Solar']
        if total_light < 30:
            SL = 0.0
        elif total_light <= 900:
            SL = 1.0
        else:
            SL = 0.7

        # P_N
        N_max = 25
        PN = max(0, abs(fitness_obj.z['N'] - N_max))

        # محاسبه p_e و سایر جریمه‌ها
        p_e = max(0, energy - 100)
        p_delta_t = abs(T_in - fitness_obj.z['T_out'])
        p_delta_h = abs(H_in - fitness_obj.z['H_out'])
        p_light_def = max(0, 500 - (L + fitness_obj.z['Solar']))
        p_n = np.sqrt(fitness_obj.z['N'])
        p_vent_risk = max(0, fitness_obj.z['N'] * fitness_obj.z['Wind'] - 50)

        f1_raw = (fitness_obj.a_norm[0] * np.log1p(p_e ** 2) +
                  fitness_obj.a_norm[1] * p_delta_t +
                  fitness_obj.a_norm[2] * p_delta_h +
                  fitness_obj.a_norm[3] * p_light_def +
                  fitness_obj.a_norm[4] * p_n +
                  fitness_obj.a_norm[5] * p_vent_risk)

        f2_raw = (fitness_obj.b_norm[0] * ST +
                  fitness_obj.b_norm[1] * SH +
                  fitness_obj.b_norm[2] * SL +
                  fitness_obj.b_norm[3] * SC -
                  fitness_obj.b_norm[4] * PN)

        f1_vals.append(f1_raw)
        f2_vals.append(f2_raw)

    return f1_vals, f2_vals


def compute_stage3_ranges(fitness_obj, pop_size=50000):
    """محاسبه بازه f1 و f2 برای مرحله 3 با استفاده از جمعیت تصادفی"""
    temp_population = []
    for _ in range(pop_size):
        individual = [
            random.uniform(*fitness_obj.bounds['T_in']),
            random.uniform(*fitness_obj.bounds['H_in']),
            random.uniform(*fitness_obj.bounds['L']),
            random.uniform(*fitness_obj.bounds['CO2'])
        ]
        temp_population.append(individual)

    f1_vals, f2_vals = calculate_raw_f1_f2_for_population(fitness_obj, temp_population)

    F1_min = np.min(f1_vals)
    F1_max = np.max(f1_vals)
    F2_min = np.min(f2_vals)
    F2_max = np.max(f2_vals)

    # اضافه کردن حاشیه 5 درصدی مثل مرحله 2 (اختیاری اما باعث پایداری می‌شود)
    margin_f1 = (F1_max - F1_min) * 0.05
    margin_f2 = (F2_max - F2_min) * 0.05
    F1_min = F1_min - margin_f1
    F1_max = F1_max + margin_f1
    F2_min = F2_min - margin_f2
    F2_max = F2_max + margin_f2

    return F1_min, F1_max, F2_min, F2_max


# ----------------اجرای مرحله سوم روی سناریوها----------------
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
    print(f"\n{'=' * 60}")
    print(f"سناریو: {sc['name']}")
    print(f"مقادیر محیطی: T_out={sc['z']['T_out']}, H_out={sc['z']['H_out']}, "
          f"Solar={sc['z']['Solar']}, Wind={sc['z']['Wind']}, N={sc['z']['N']}, W={sc['z']['W']}")
    print(f"{'=' * 60}")

    # ابتدا یک شیء Fitness موقت برای محاسبه بازه ایجاد می‌کنیم
    temp_fitness_obj = Fitness(best_chrom, sc['z'], sc['bounds'], 0, 1, 0, 1)

    # محاسبه بازه واقعی f1 و f2 برای این سناریو
    F1_min, F1_max, F2_min, F2_max = compute_stage3_ranges(temp_fitness_obj, pop_size=50000)

    print(f"بازه محاسبه شده برای این سناریو:")
    print(f"  F1: [{F1_min:.6f}, {F1_max:.6f}]")
    print(f"  F2: [{F2_min:.6f}, {F2_max:.6f}]")

    # ایجاد شیء Fitness اصلی با بازه‌های صحیح
    fitness_obj = Fitness(best_chrom, sc['z'], sc['bounds'], F1_min, F1_max, F2_min, F2_max)

    # ==========  (Best Cost و Avg Cost) ==========
    fig_cost, axes_cost = plt.subplots(2, 2, figsize=(14, 10))
    axes_cost = axes_cost.flatten()

    # ==========  f1_norm و f2_norm ==========
    fig_f, axes_f = plt.subplots(2, 2, figsize=(14, 10))
    axes_f = axes_f.flatten()

    for idx, cfg in enumerate(configs):
        print(f"\n اجرای {cfg['name']}...")

        ga = GA(fitness_obj,
                pop_size=100,
                generations=200,
                crossover_rate=0.9,
                mutation_rate=cfg['mut'],
                selection_method=cfg['selection'],
                crossover_method=cfg['crossover'],
                tournament_size=3)

        # دریافت تاریخچه هزینه و f1_norm و f2_norm
        best_ind, best_cost, best_hist, avg_hist, f1_norm_hist, f2_norm_hist = ga.run()


        ax_cost = axes_cost[idx]
        ax_cost.plot(best_hist, label='Best Cost', color='blue', linewidth=2)
        ax_cost.plot(avg_hist, label='Avg Cost', color='red', linestyle='--', linewidth=2)
        ax_cost.set_title(f"{cfg['name']}\nBest Cost = {best_cost:.4f}", fontsize=10)
        ax_cost.set_xlabel('Generation')
        ax_cost.set_ylabel('Cost')
        ax_cost.legend(loc='upper right', fontsize=8)
        ax_cost.grid(True, alpha=0.3)


        ax_f = axes_f[idx]
        ax_f.plot(f1_norm_hist, label='f1_norm (Energy Cost) - lower is better', color='green', linewidth=2)
        ax_f.plot(f2_norm_hist, label='f2_norm (Growth Quality) - higher is better', color='orange', linewidth=2)
        ax_f.set_title(
            f"{cfg['name']}\nf1_norm_final = {f1_norm_hist[-1]:.4f} , f2_norm_final = {f2_norm_hist[-1]:.4f}",
            fontsize=10)
        ax_f.set_xlabel('Generation')
        ax_f.set_ylabel('Normalized Value (0 to 1)')
        ax_f.legend(loc='upper right', fontsize=8)
        ax_f.grid(True, alpha=0.3)
        ax_f.set_ylim(-0.1, 1.1)  # محدوده 0 تا 1 برای نمایش بهتر

        print(f"    بهترین هزینه: {best_cost:.6f}")
        print(f"    f1_norm نهایی: {f1_norm_hist[-1]:.6f} (هرچه کمتر بهتر)")
        print(f"    f2_norm نهایی: {f2_norm_hist[-1]:.6f} (هرچه بیشتر بهتر)")
        print(f"    بهترین ورودی: T_in={best_ind[0]:.2f}, H_in={best_ind[1]:.2f}, "
              f"L={best_ind[2]:.2f}, CO2={best_ind[3]:.2f}")

    fig_cost.suptitle(f"Scenario: {sc['name']} - Cost Convergence", fontsize=14)
    fig_cost.tight_layout()

    fig_f.suptitle(f"Scenario: {sc['name']} - f1_norm & f2_norm Convergence", fontsize=14)
    fig_f.tight_layout()

    plt.show()

print("\n" + "=" * 60)
print("اجرای تمام سناریوها به پایان رسید.")
print("=" * 60)