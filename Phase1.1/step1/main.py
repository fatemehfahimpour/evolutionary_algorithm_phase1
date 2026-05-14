import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Preprocessing import get_preprocessed_data
from genetic_algorithm import run_ga
from Fitness_function import fitness, calculate_f1_score, calculate_f2_score, normalize_coefficients

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

#------------------تحلیل حساسیت ضرایب--------------------
best_result = max(results, key=lambda x: x["best_fit"])
print("\n" + "=" * 60)
print(f"بهترین ترکیب: {best_result['name']} با Fitness = {best_result['best_fit']:.6f}")
print("=" * 60)

best_chromosome = best_result["best_chrom"]

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


perturb = 0.05
df_sensitivity, base_fit, base_f1, base_f2 = sensitivity_analysis(df_clean, best_chromosome, perturbation=perturb)

print("\n--- تحلیل حساسیت (افزایش {:.0%} هر ضریب) ---".format(perturb))
print(f"مقادیر پایه: Fitness = {base_fit:.6f}, f1_score = {base_f1:.6f}, f2_score = {base_f2:.6f}\n")
print(df_sensitivity.to_string(index=False, float_format="%.6f"))


plt.figure(figsize=(12, 6))
plt.bar(df_sensitivity['Coefficient'], df_sensitivity['ΔFitness'], color='steelblue')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('ضریب')
plt.ylabel('تغییر در Fitness')
plt.title(f'تحلیل حساسیت: اثر افزایش {perturb:.0%} هر ضریب روی Fitness')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


x = np.arange(len(df_sensitivity['Coefficient']))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width / 2, df_sensitivity['Δf1_score'], width, label='Δf1_score', color='green')
ax.bar(x + width / 2, df_sensitivity['Δf2_score'], width, label='Δf2_score', color='orange')
ax.set_xlabel('ضریب')
ax.set_ylabel('تغییر امتیاز')
ax.set_title(f'تأثیر افزایش {perturb:.0%} هر ضریب روی f1_score و f2_score')
ax.set_xticks(x)
ax.set_xticklabels(df_sensitivity['Coefficient'], rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# #---------------------حلیل رفتار مدل در نقاط بحرانی دیتاست---------------------
print("\n" + "=" * 80)
print("تحلیل رفتار مدل در نقاط بحرانی دیتاست")
print("=" * 80)


# با استفاده از بهترین کروموزوم (best_chromosome)
a_raw = best_chromosome[:6]
b_raw = best_chromosome[6:]
a_norm = normalize_coefficients(a_raw)
b_norm = normalize_coefficients(b_raw)

f1_raw_all = []
f2_raw_all = []
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
    f1_raw_all.append(f1_raw)

    # محاسبه f2_raw
    f2_raw = (b_norm[0] * row['S_T'] +
              b_norm[1] * row['S_H'] +
              b_norm[2] * row['S_L'] +
              b_norm[3] * row['S_C'] -
              b_norm[4] * row['P_N'])
    f2_raw_all.append(f2_raw)

F1_min = np.min(f1_raw_all)
F1_max = np.max(f1_raw_all)
F2_min = np.min(f2_raw_all)
F2_max = np.max(f2_raw_all)
delta = 1e-8

print(f"بازه f1_raw: [{F1_min:.3f}, {F1_max:.3f}]")
print(f"بازه f2_raw: [{F2_min:.3f}, {F2_max:.3f}]")


def compute_f1_score_fixed(data_row, a_norm):
    # data_row: دیکشنری یا سری شامل مقادیر مورد نیاز
    p_e = max(0, data_row['E'] - 100)
    p_delta_t = abs(data_row['T_in'] - data_row['T_out'])
    p_delta_h = abs(data_row['H_in'] - data_row['H_out'])
    p_light_def = max(0, 500 - (data_row['L'] + data_row['Solar']))
    p_n = np.sqrt(data_row['N'])
    p_vent_risk = max(0, data_row['N'] * data_row['Wind'] - 50)
    f1_raw = (a_norm[0] * np.log1p(p_e ** 2) +
              a_norm[1] * p_delta_t +
              a_norm[2] * p_delta_h +
              a_norm[3] * p_light_def +
              a_norm[4] * p_n +
              a_norm[5] * p_vent_risk)
    f1_norm = (f1_raw - F1_min) / (F1_max - F1_min + delta)
    f1_norm = np.clip(f1_norm, 0, 1)
    return 1 - f1_norm  # چون f1 کمینه شود


def compute_f2_score_fixed(data_row, b_norm):
    f2_raw = (b_norm[0] * data_row['S_T'] +
              b_norm[1] * data_row['S_H'] +
              b_norm[2] * data_row['S_L'] +
              b_norm[3] * data_row['S_C'] -
              b_norm[4] * data_row['P_N'])
    f2_norm = (f2_raw - F2_min) / (F2_max - F2_min + delta)
    f2_norm = np.clip(f2_norm, 0, 1)
    return f2_norm  # f2 بیشینه شود

def energy_single(T_in, T_out, H_in, L, N, CO2, weather='sunny'):
    k_w_map = {"night": 0.8, "sunny": 0.9, "cloudy": 1.0, "humid": 1.08,
               "rainy": 1.15, "stormy": 1.25, "cold": 1.3}
    k_w = k_w_map.get(weather, 0.9)
    delta_T = abs(T_in - T_out)
    P_hum = max(0, H_in - 60) ** 2 + max(0, 30 - H_in) ** 2
    energy = k_w * (0.8 * delta_T ** 2 + 12 * np.log(1 + N) + 0.02 * P_hum)
    energy += 6 * np.log(1 + L) + 2 * np.log(1 + CO2)
    return energy

mean_vals = {
    'T_in': df_clean['T_in'].mean(),
    'T_out': df_clean['T_out'].mean(),
    'H_in': df_clean['H_in'].mean(),
    'H_out': df_clean['H_out'].mean() if 'H_out' in df_clean else df_clean['H_in'].mean() - 5,
    'L': df_clean['L'].mean(),
    'Solar': df_clean['Solar'].mean(),
    'CO2': df_clean['CO2'].mean(),
    'N': int(round(df_clean['N'].mean())),
    'Wind': df_clean['Wind'].mean()
}

scenarios = {
    "مرجع (میانگین دیتاست)": mean_vals,
    "ایده‌آل (بهترین رشد)": {"T_in": 22, "T_out": 20, "H_in": 50, "H_out": 45, "L": 400, "Solar": 200, "CO2": 1000,
                             "N": 10, "Wind": 5},
    "گرمای شدید (داخل و بیرون گرم)": {"T_in": 35, "T_out": 38, "H_in": 40, "H_out": 35, "L": 500, "Solar": 600,
                                      "CO2": 800, "N": 10, "Wind": 5},
    "سرمای شدید (داخل و بیرون سرد)": {"T_in": 10, "T_out": 5, "H_in": 50, "H_out": 45, "L": 300, "Solar": 100,
                                      "CO2": 1000, "N": 10, "Wind": 5},
    "رطوبت بسیار بالا (داخل مرطوب)": {"T_in": 24, "T_out": 22, "H_in": 85, "H_out": 80, "L": 400, "Solar": 200,
                                      "CO2": 900, "N": 10, "Wind": 5},
    "رطوبت بسیار کم (داخل خشک)": {"T_in": 24, "T_out": 22, "H_in": 15, "H_out": 12, "L": 400, "Solar": 200, "CO2": 900,
                                  "N": 10, "Wind": 5},
    "نور بسیار کم (شب ابری)": {"T_in": 22, "T_out": 20, "H_in": 50, "H_out": 45, "L": 0, "Solar": 0, "CO2": 900,
                               "N": 10, "Wind": 5},
    "نور بسیار زیاد (ظهر آفتابی)": {"T_in": 25, "T_out": 28, "H_in": 55, "H_out": 50, "L": 1000, "Solar": 800,
                                    "CO2": 1100, "N": 10, "Wind": 5},
    "تراکم بسیار بالا (ازدحام گیاه)": {"T_in": 24, "T_out": 22, "H_in": 60, "H_out": 55, "L": 500, "Solar": 300,
                                       "CO2": 1200, "N": 30, "Wind": 5},
    "CO₂ بسیار کم (کمبود دی‌اکسید)": {"T_in": 22, "T_out": 20, "H_in": 50, "H_out": 45, "L": 400, "Solar": 200,
                                      "CO2": 400, "N": 10, "Wind": 5},
    "CO₂ بسیار زیاد (غلظت سمی)": {"T_in": 22, "T_out": 20, "H_in": 50, "H_out": 45, "L": 400, "Solar": 200, "CO2": 1600,
                                  "N": 10, "Wind": 5},
    "باد شدید (تهویه زیاد)": {"T_in": 22, "T_out": 20, "H_in": 50, "H_out": 45, "L": 400, "Solar": 200, "CO2": 900,
                              "N": 10, "Wind": 20}
}


def evaluate_scenario_fixed(scenario):
    E = energy_single(scenario['T_in'], scenario['T_out'], scenario['H_in'],
                      scenario['L'], scenario['N'], scenario['CO2'], weather='sunny')

    def S_T(t):
        if 20 <= t <= 24: return 1
        if 18 <= t < 20 or 24 < t <= 26: return 0.6
        if 16 <= t < 18 or 26 < t <= 28: return 0.2
        return 0

    def S_H(h):
        if 45 <= h <= 60: return 1
        if 35 <= h < 45 or 60 < h <= 70: return 0.5
        return 0

    def S_C(c):
        if 800 <= c <= 1200: return 1
        if 700 <= c < 800 or 1200 < c <= 1400: return 0.4
        return 0

    total_light = scenario['L'] + scenario['Solar']
    if total_light < 30:
        SL = 0.0
    elif total_light <= 900:
        SL = 1.0
    else:
        SL = 0.7
    PN = max(0, scenario['N'] - 25)

    data_row = {
        'E': E,
        'T_in': scenario['T_in'],
        'T_out': scenario['T_out'],
        'H_in': scenario['H_in'],
        'H_out': scenario.get('H_out', scenario['H_in'] - 5),
        'L': scenario['L'],
        'Solar': scenario['Solar'],
        'CO2': scenario['CO2'],
        'Wind': scenario['Wind'],
        'N': scenario['N'],
        'S_T': S_T(scenario['T_in']),
        'S_H': S_H(scenario['H_in']),
        'S_C': S_C(scenario['CO2']),
        'S_L': SL,
        'P_N': PN
    }

    f1_sc = compute_f1_score_fixed(data_row, a_norm)
    f2_sc = compute_f2_score_fixed(data_row, b_norm)
    rx = (0.1) * (-np.sum(a_norm * np.log(a_norm + 1e-12)) / np.log(6) +
                  -np.sum(b_norm * np.log(b_norm + 1e-12)) / np.log(5))
    fitness_val = 0.4 * f1_sc + 0.6 * f2_sc + rx
    return f1_sc, f2_sc, fitness_val


# اجرا روی همه سناریوها
results_critical = []
for name, scen in scenarios.items():
    f1, f2, fit = evaluate_scenario_fixed(scen)
    results_critical.append({
        'Scenario': name,
        'f1_score': f1,
        'f2_score': f2,
        'Fitness': fit
    })

df_critical = pd.DataFrame(results_critical)
print("\nنتایج تحلیل نقاط بحرانی")
print(df_critical.to_string(index=False, float_format="%.4f"))
