import numpy as np
from step1.Preprocessing import get_k_w

def fitness(a_norm, b_norm , df):
    WEATHER_LIST = ["night", "sunny", "cloudy", "humid", "rainy", "stormy", "cold"]

    def calculate_energy(individual):
        T_in, T_out, H_in, L, N, CO2, wind, solar, H_out = individual[:9]
        weather_vec = individual[9:]
        if 1 in weather_vec:
            weather_idx = weather_vec.index(1)
            weather = WEATHER_LIST[weather_idx]
        else:
            weather = 'sunny'
        weather_dict = {f'W_{weather}': 1}
        k_w = get_k_w(weather_dict)
        delta_T = abs(T_in - T_out)
        P_hum = max(0, H_in - 60)**2 + max(0, 30 - H_in)**2
        energy = k_w * (0.8 * delta_T**2 + 12 * np.log(1 + N) + 0.02 * P_hum)
        energy += 6 * np.log(1 + L) + 2 * np.log(1 + CO2)
        return energy

    def calculate_comfort(individual):
        T_in = individual[0]
        H_in = individual[2]
        L = individual[3]
        CO2 = individual[5]
        N = individual[4]
        Solar = individual[7]

        # S_T
        if 20 <= T_in <= 24: ST = 1
        elif 18 <= T_in < 20 or 24 < T_in <= 26: ST = 0.6
        elif 16 <= T_in < 18 or 26 < T_in <= 28: ST = 0.2
        else: ST = 0

        # S_H
        if 45 <= H_in <= 60: SH = 1
        elif 35 <= H_in < 45 or 60 < H_in <= 70: SH = 0.5
        else: SH = 0

        # S_C
        if 800 <= CO2 <= 1200: SC = 1
        elif 700 <= CO2 < 800 or 1200 < CO2 <= 1400: SC = 0.4
        else: SC = 0
        total_light = L + Solar
        if total_light < 30: SL = 0.0
        elif total_light <= 900: SL = 1.0
        else: SL = 0.7
        PN = max(0, abs(N - 30))
        f2_raw = (b_norm[0]*ST + b_norm[1]*SH + b_norm[2]*SL +
                  b_norm[3]*SC - b_norm[4]*PN)
        # نرمال‌سازی به [0,1] با فرض f2_raw در [-1, 1]
        return np.clip((f2_raw + 1) / 2, 0, 1)

    def constraint_penalty(individual,df):
        T_in, T_out, H_in, L, N, CO2, wind, solar, H_out = individual[:9]
        penalty = 0
        if T_in < 18 or T_in > 30: penalty += 50
        if T_out < 0 or T_out > 40: penalty += 50
        if H_in < 20 or H_in > 80: penalty += 50
        if L < 100 or L > 900: penalty += 50
        if N < 1 or N > 30: penalty += 100
        if CO2 < 400 or CO2 > 1500: penalty += 100
        if wind < df['Wind'].min() or wind > df['Wind'].max(): penalty += 50
        if solar < df['Solar'].min() or solar > df['Solar'].max(): penalty += 50
        if H_out < df['H_out'].min() or H_out > df['H_out'].max(): penalty += 50
        return penalty

    def fitness_func(individual, lambda_real=0.0, real_penalty=None):
        E = calculate_energy(individual)
        # نرمال‌سازی انرژی با بازه تخمینی 0 تا 2500 (بر اساس دیتاست)
        E_norm = np.clip(E / 2500.0, 0, 1)
        energy_score = 1 - E_norm
        comfort_score = calculate_comfort(individual)
        base_fitness = 0.4 * energy_score + 0.6 * comfort_score
        penalty = constraint_penalty(individual,df)
        if real_penalty is not None:
            penalty += lambda_real * real_penalty(individual)
        fitness = base_fitness - penalty / 100.0   # کاهش تأثیر جریمه
        return fitness

    return fitness_func