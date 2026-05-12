import numpy as np


def calculate_f1_score(df, a_norm_coefficients):
    # in chromosome first 6 elements are coefficients for f1 function
    a1_norm, a2_norm, a3_norm, a4_norm, a5_norm, a6_norm = a_norm_coefficients

    p_e = np.maximum(0, df['E'] - 100)
    p_delta_t = np.abs(df['T_in'] - df['T_out'])
    p_delta_h = np.abs(df['H_in'] - df['H_out'])
    p_light_def = np.maximum(0, 500 - (df['L'] + df['Solar']))
    p_n = np.sqrt(df['N'])
    p_vent_risk = np.maximum(0, df['N'] * df['Wind'] - 50)

    f1_raw = a1_norm * np.log1p(
        p_e ** 2) + a2_norm * p_delta_t + a3_norm * p_delta_h + a4_norm * p_light_def + a5_norm * p_n + a6_norm * p_vent_risk
    f1_min = f1_raw.min()
    f1_max = f1_raw.max()

    # جلوگیری از تقسیم بر صفر
    if f1_max == f1_min:
        return f1_raw * 0

    epsilon = 1e-8
    normalized_f1 = (f1_raw - f1_min) / ((f1_max - f1_min) + epsilon)

    return 1 - normalized_f1


def calculate_f2_score(data, b_norm_coefficients):
    # in chromosome last 5 elements are coefficients for f2 function
    b1, b2, b3, b4, b5 = b_norm_coefficients

    f2_raw = (
            b1 * data['S_T'] +
            b2 * data['S_H'] +
            b3 * data['S_L'] +
            b4 * data['S_C'] -
            b5 * data['P_N']
    )

    f2_min = f2_raw.min()
    f2_max = f2_raw.max()

    normalized_f2 = (f2_raw - f2_min) / (f2_max - f2_min + 1e-8)

    return normalized_f2


def calculate_RX(a_norm_coefficients, b_norm_coefficients):
    a = np.array(a_norm_coefficients)
    b = np.array(b_norm_coefficients)

    epsilon = 1e-12
    h_a = -np.sum(a * np.log(a + epsilon))
    h_b = -np.sum(b * np.log(b + epsilon))

    h_norm_a = h_a / np.log(6)
    h_norm_b = h_b / np.log(5)

    lambd = 0.1
    r_x = lambd * (h_norm_a + h_norm_b)

    return r_x


def normalize_coefficients(coefficients, epsilon=1e-12):
    coefficients = np.array(coefficients, dtype=float)
    total = np.sum(coefficients) + epsilon
    if total < epsilon:
        return np.ones_like(coefficients) / len(coefficients)
    return (coefficients + epsilon) / total


def fitness(df_clean, chromosome):
    # chromosome: [a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5]
    a_coefficients = chromosome[:6]
    b_coefficients = chromosome[6:]

    a_norm = normalize_coefficients(a_coefficients)
    b_norm = normalize_coefficients(b_coefficients)

    f1_score = calculate_f1_score(df_clean, a_norm)
    f2_score = calculate_f2_score(df_clean, b_norm)
    rx_score = calculate_RX(a_norm, b_norm)

    fitness_value = 0.4 * f1_score.mean() + 0.6 * f2_score.mean() + rx_score

    return fitness_value