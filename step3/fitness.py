import numpy as np
from step1.Fitness_function import normalize_coefficients


class Fitness:
    def __init__(self, best_chromosome, z, bounds, F1_min, F1_max, F2_min, F2_max):
        self.best_chromosome = best_chromosome
        self.z = z  # T_out ,H_out ,Solar,Wind,N,W
        self.bounds = bounds
        self.F1_MIN = F1_min
        self.F1_MAX = F1_max
        self.F2_MIN = F2_min
        self.F2_MAX = F2_max
        self.delta = 1e-8

        ai = best_chromosome[:6]
        bj = best_chromosome[6:]
        self.a_norm = normalize_coefficients(ai)
        self.b_norm = normalize_coefficients(bj)

        epsilon = 1e-12
        h_a = -np.sum(self.a_norm * np.log(self.a_norm + epsilon))
        h_b = -np.sum(self.b_norm * np.log(self.b_norm + epsilon))
        h_norm_a = h_a / np.log(6)
        h_norm_b = h_b / np.log(5)
        self.rx = 0.1 * (h_norm_a + h_norm_b)

        self.weather = self.get_k_w(z['W'])

    def get_k_w(self, weather):
        k_w_map = {"night": 0.8, "sunny": 0.9, "cloudy": 1.0, "humid": 1.08,
                   "rainy": 1.15, "stormy": 1.25, "cold": 1.3}
        return k_w_map.get(weather, 1.0)  # اگر هوا هیچ یک از مقادیر بالا نباشد مقدار 1.0 را برمیگرداند.

    def calculate_energy(self, u):
        T_in, H_in, L, CO2 = u
        delta_T = abs(T_in - self.z['T_out'])
        P_hum = max(0, H_in - 60) ** 2 + max(0, 30 - H_in) ** 2
        energy = self.weather * (0.8 * delta_T ** 2 + 12 * np.log(1 + self.z['N']) + 0.02 * P_hum)
        energy += 6 * np.log(1 + L) + 2 * np.log(1 + CO2)
        return energy

    def calculate_quality_scores(self, u):
        T_in, H_in, L, CO2 = u

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
        total_light = L + self.z['Solar']
        if total_light < 30:
            SL = 0.0
        elif total_light <= 900:
            SL = 1.0
        else:
            SL = 0.7

        # P_N
        N_max = 25
        PN = max(0, abs(self.z['N'] - N_max))
        return ST, SH, SL, SC, PN

    def calculate_cost(self, u):
        E = self.calculate_energy(u)
        ST, SH, SL, SC, PN = self.calculate_quality_scores(u)

        p_e = max(0, E - 100)
        p_delta_t = abs(u[0] - self.z['T_out'])
        p_delta_h = abs(u[1] - self.z['H_out'])
        p_light_def = max(0, 500 - (u[2] + self.z['Solar']))
        p_n = np.sqrt(self.z['N'])
        p_vent_risk = max(0, self.z['N'] * self.z['Wind'] - 50)

        f1 = (self.a_norm[0] * np.log1p(p_e ** 2) +
              self.a_norm[1] * p_delta_t +
              self.a_norm[2] * p_delta_h +
              self.a_norm[3] * p_light_def +
              self.a_norm[4] * p_n +
              self.a_norm[5] * p_vent_risk)

        f2 = (self.b_norm[0] * ST +
              self.b_norm[1] * SH +
              self.b_norm[2] * SL +
              self.b_norm[3] * SC -
              self.b_norm[4] * PN)

        # نرمال‌سازی با بازه‌های ثابت و جریمه برای خروج از بازه
        if f1 <= self.F1_MIN:
            f1_norm = 0.0
            penalty1 = 0.0
        elif f1 >= self.F1_MAX:
            f1_norm = 1.0
            penalty1 = (f1 - self.F1_MAX) / (self.F1_MAX - self.F1_MIN + self.delta)
        else:
            f1_norm = (f1 - self.F1_MIN) / (self.F1_MAX - self.F1_MIN + self.delta)
            penalty1 = 0.0

        if f2 <= self.F2_MIN:
            f2_norm = 0.0
            penalty2 = (self.F2_MIN - f2) / (self.F2_MAX - self.F2_MIN + self.delta)
        elif f2 >= self.F2_MAX:
            f2_norm = 1.0
            penalty2 = 0.0
        else:
            f2_norm = (f2 - self.F2_MIN) / (self.F2_MAX - self.F2_MIN + self.delta)
            penalty2 = 0.0

        f1_score = 1 - f1_norm
        f2_score = f2_norm
        total_penalty = 0.1 * (penalty1 + penalty2)
        fitness = 0.4 * f1_score + 0.6 * f2_score + self.rx - total_penalty
        cost = -fitness

        # برگرداندن cost به همراه f1_norm و f2_norm
        return cost, f1_norm, f2_norm

    def calculate_raw_f1_f2(self, u):
        """محاسبه مقادیر خام f1 و f2 بدون نرمال‌سازی - برای محاسبه بازه مرحله 3"""
        E = self.calculate_energy(u)
        ST, SH, SL, SC, PN = self.calculate_quality_scores(u)

        p_e = max(0, E - 100)
        p_delta_t = abs(u[0] - self.z['T_out'])
        p_delta_h = abs(u[1] - self.z['H_out'])
        p_light_def = max(0, 500 - (u[2] + self.z['Solar']))
        p_n = np.sqrt(self.z['N'])
        p_vent_risk = max(0, self.z['N'] * self.z['Wind'] - 50)

        f1_raw = (self.a_norm[0] * np.log1p(p_e ** 2) +
                  self.a_norm[1] * p_delta_t +
                  self.a_norm[2] * p_delta_h +
                  self.a_norm[3] * p_light_def +
                  self.a_norm[4] * p_n +
                  self.a_norm[5] * p_vent_risk)

        f2_raw = (self.b_norm[0] * ST +
                  self.b_norm[1] * SH +
                  self.b_norm[2] * SL +
                  self.b_norm[3] * SC -
                  self.b_norm[4] * PN)

        return f1_raw, f2_raw
