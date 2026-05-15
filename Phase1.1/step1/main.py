import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import illustration
from Preprocessing import get_preprocessed_data
from genetic_algorithm import run_ga
from Fitness_function import fitness, calculate_f1_score, calculate_f2_score, normalize_coefficients

DATA_PATH = 'data/1_26336110128.csv'
df_clean = get_preprocessed_data(DATA_PATH)
delta = 1e-8

results = illustration.different_combination(df_clean)

best_chromosome = illustration.coefficients_sensitive_analysis(df_clean, results)

F1_min, F1_max, F2_min, F2_max, a_norm, b_norm = illustration.critical_points_analysis(df_clean, best_chromosome)


def compute_f1_score_fixed(data_row, a_norm):
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
    return 1 - f1_norm


def compute_f2_score_fixed(data_row, b_norm):
    f2_raw = (b_norm[0] * data_row['S_T'] +
              b_norm[1] * data_row['S_H'] +
              b_norm[2] * data_row['S_L'] +
              b_norm[3] * data_row['S_C'] -
              b_norm[4] * data_row['P_N'])
    f2_norm = (f2_raw - F2_min) / (F2_max - F2_min + delta)
    f2_norm = np.clip(f2_norm, 0, 1)
    return f2_norm


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
    "Reference (Dataset Mean)": mean_vals,

    "Ideal (Best Growth Conditions)": {
        "T_in": 22,
        "T_out": 20,
        "H_in": 50,
        "H_out": 45,
        "L": 400,
        "Solar": 200,
        "CO2": 1000,
        "N": 10,
        "Wind": 5
    },

    "Extreme Heat (Hot Indoor and Outdoor)": {
        "T_in": 35,
        "T_out": 38,
        "H_in": 40,
        "H_out": 35,
        "L": 500,
        "Solar": 600,
        "CO2": 800,
        "N": 10,
        "Wind": 5
    },

    "Extreme Cold (Cold Indoor and Outdoor)": {
        "T_in": 10,
        "T_out": 5,
        "H_in": 50,
        "H_out": 45,
        "L": 300,
        "Solar": 100,
        "CO2": 1000,
        "N": 10,
        "Wind": 5
    },

    "Very High Humidity (Humid Indoor)": {
        "T_in": 24,
        "T_out": 22,
        "H_in": 85,
        "H_out": 80,
        "L": 400,
        "Solar": 200,
        "CO2": 900,
        "N": 10,
        "Wind": 5
    },

    "Very Low Humidity (Dry Indoor)": {
        "T_in": 24,
        "T_out": 22,
        "H_in": 15,
        "H_out": 12,
        "L": 400,
        "Solar": 200,
        "CO2": 900,
        "N": 10,
        "Wind": 5
    },

    "Very Low Light (Cloudy Night)": {
        "T_in": 22,
        "T_out": 20,
        "H_in": 50,
        "H_out": 45,
        "L": 0,
        "Solar": 0,
        "CO2": 900,
        "N": 10,
        "Wind": 5
    },

    "Very High Light (Sunny Noon)": {
        "T_in": 25,
        "T_out": 28,
        "H_in": 55,
        "H_out": 50,
        "L": 1000,
        "Solar": 800,
        "CO2": 1100,
        "N": 10,
        "Wind": 5
    },

    "Very High Density (Plant Overcrowding)": {
        "T_in": 24,
        "T_out": 22,
        "H_in": 60,
        "H_out": 55,
        "L": 500,
        "Solar": 300,
        "CO2": 1200,
        "N": 30,
        "Wind": 5
    },

    "Very Low CO2 (Carbon Dioxide Deficiency)": {
        "T_in": 22,
        "T_out": 20,
        "H_in": 50,
        "H_out": 45,
        "L": 400,
        "Solar": 200,
        "CO2": 400,
        "N": 10,
        "Wind": 5
    },

    "Very High CO2 (Toxic Concentration)": {
        "T_in": 22,
        "T_out": 20,
        "H_in": 50,
        "H_out": 45,
        "L": 400,
        "Solar": 200,
        "CO2": 1600,
        "N": 10,
        "Wind": 5
    },

    "Strong Wind (High Ventilation)": {
        "T_in": 22,
        "T_out": 20,
        "H_in": 50,
        "H_out": 45,
        "L": 400,
        "Solar": 200,
        "CO2": 900,
        "N": 10,
        "Wind": 20
    }
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
print("\nresult of critical points analysis")
print(df_critical.to_string(index=False, float_format="%.4f"))
