# chromosome = [T_in, T_out, H_in, L, N, CO2, wind, solar, H_out] + weather_vector
import numpy as np

from Preprocessing import get_k_w


def calculate_f1(chromosome, a, tau=50):
    T_in, T_out, H_in, L, N, CO2, wind, solar, H_out = chromosome[:9]

    # ---------- energy model ----------

    delta_T = abs(T_in - T_out)

    P_hum = max(0, H_in - 60) ** 2 + max(0, 30 - H_in) ** 2

    # weather factor
    WEATHER_LIST = ["night", "sunny", "cloudy", "humid", "rainy", "stormy", "cold"]

    weather_vec = chromosome[9:]
    if 1 in weather_vec:
        weather_idx = weather_vec.index(1)
        weather = WEATHER_LIST[weather_idx]
    else:
        weather = 'sunny'
    weather_dict = {f'W_{weather}': 1}
    k_w = get_k_w(weather_dict)

    E = (
            k_w * (0.8 * delta_T + 12 * np.log(1 + N) + 0.02 * P_hum)
            + 6 * np.log(1 + L)
            + 2 * np.log(1 + CO2)
    )

    # ---------- penalties ----------

    P_E = max(0, E - 100)

    P_dT = abs(T_in - T_out)

    P_dH = abs(H_in - H_out)

    L_opt = 500
    P_light = max(0, L_opt - (L + solar))

    P_vent = max(0, N * wind - tau)

    # ---------- final f1 ----------

    a1, a2, a3, a4, a5, a6 = a

    f1 = (
            a1 * np.log(1 + P_E ** 2)
            + a2 * P_dT
            + a3 * P_dH
            + a4 * P_light
            + a5 * np.sqrt(N)
            + a6 * P_vent
    )

    return f1


def calculate_f2(chromosome, b, n_max=25):
    T_in, T_out, H_in, L, N, CO2, wind, solar, H_out = chromosome[:9]
    # -------- temperature score --------
    if 20 <= T_in <= 24:
        s_t = 1
    elif 18 <= T_in < 20 or 24 < T_in <= 26:
        s_t = 0.6
    elif 16 <= T_in < 18 or 26 < T_in <= 28:
        s_t = 0.2
    else:
        s_t = 0

    # -------- humidity score --------
    if 45 <= H_in <= 60:
        s_h = 1
    elif 35 <= H_in < 45 or 60 < H_in <= 70:
        s_h = 0.5
    else:
        s_h = 0

    # -------- light score --------
    l_min = 30
    l_max = 900

    light_total = L + solar

    if light_total < l_min:
        s_l = 0
    elif l_min <= light_total <= l_max:
        s_l = 1
    else:
        s_l = 0.7

    # -------- CO2 score --------
    if 800 <= CO2 <= 1200:
        s_c = 1
    elif 700 <= CO2 < 800 or 1200 < CO2 <= 1400:
        s_c = 0.4
    else:
        s_c = 0

    # -------- density penalty --------
    p_n = max(0, abs(N - n_max))

    b1, b2, b3, b4, b5 = b

    f2 = (
            b1 * s_t +
            b2 * s_h +
            b3 * s_l +
            b4 * s_c -
            b5 * p_n
    )

    return f2


def compute_objective_ranges(population, a, b):
    f1_vals = [calculate_f1(ch, a) for ch in population]
    f2_vals = [calculate_f2(ch, b) for ch in population]

    F1_min = min(f1_vals)
    F1_max = max(f1_vals)
    F2_min = min(f2_vals)
    F2_max = max(f2_vals)

    return F1_min, F1_max, F2_min, F2_max


def fitness_function(chromosome, a, b, F1_min, F1_max, F2_min, F2_max, w1=0.4, w2=0.6, delta=1e-6):
    f1 = calculate_f1(chromosome, a)
    f2 = calculate_f2(chromosome, b)

    f1_norm = (f1 - F1_min) / (F1_max - F1_min + delta)
    f2_norm = (f2 - F2_min) / (F2_max - F2_min + delta)

    f1_norm = np.clip(f1_norm, 0, 1)
    f2_norm = np.clip(f2_norm, 0, 1)

    f1_norm = 1 - f1_norm

    fitness = w1 * f1_norm + w2 * f2_norm

    return fitness
