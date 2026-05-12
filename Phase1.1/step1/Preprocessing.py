import pandas as pd
import numpy as np


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    # Remove Missing Values
    essential_cols = ['T_in', 'T_out', 'H_in', 'H_out', 'L', 'Solar', 'CO2', 'Wind', 'N', 'W', 'E']
    df = df.dropna(subset=essential_cols)

    df = df[df['L'] >= 0]
    df = df[df['CO2'] >= 0]
    df = df[df['N'] >= 0]
    # Remove Outliers
    Q1 = df['E'].quantile(0.25)
    Q3 = df['E'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['E'] >= lower_bound) & (df['E'] <= upper_bound)]
    df = df[df['E'] >= 0]
    return df


def one_hot_encode_weather(df):
    return pd.get_dummies(df, columns=['W'], prefix='W', dtype=int)


def get_k_w(row):
    if row.get('W_night', 0) == 1:
        return 0.8
    elif row.get('W_sunny', 0) == 1:
        return 0.9
    elif row.get('W_cloudy', 0) == 1:
        return 1.0
    elif row.get('W_humid', 0) == 1:
        return 1.08
    elif row.get('W_rainy', 0) == 1:
        return 1.15
    elif row.get('W_stormy', 0) == 1:
        return 1.25
    elif row.get('W_cold', 0) == 1:
        return 1.3
    else:
        return 1.0


def calculate_energy_column(df):
    def calculate_energy(row):
        delta_T = abs(row['T_in'] - row['T_out'])
        P_hum = max(0, row['H_in'] - 60) ** 2 + max(0, 30 - row['H_in']) ** 2
        k_w = get_k_w(row)
        energy = k_w * (0.8 * delta_T ** 2 + 12 * np.log(1 + row['N']) + 0.02 * P_hum)
        energy += 6 * np.log(1 + row['L']) + 2 * np.log(1 + row['CO2'])
        return energy

    df['E_calculated'] = df.apply(calculate_energy, axis=1)
    return df


def calculate_quality_scores(df):
    def S_T(T_in):
        if 20 <= T_in <= 24:
            return 1
        elif 18 <= T_in < 20 or 24 < T_in <= 26:
            return 0.6
        elif 16 <= T_in < 18 or 26 < T_in <= 28:
            return 0.2
        return 0

    def S_H(H_in):
        if 45 <= H_in <= 60:
            return 1
        elif 35 <= H_in < 45 or 60 < H_in <= 70:
            return 0.5
        return 0

    def S_C(CO2):
        if 800 <= CO2 <= 1200:
            return 1
        elif 700 <= CO2 < 800 or 1200 < CO2 <= 1400:
            return 0.4
        return 0

    L_min, L_sat = 30, 900

    df['S_T'] = df['T_in'].apply(S_T)
    df['S_H'] = df['H_in'].apply(S_H)
    df['S_C'] = df['CO2'].apply(S_C)

    total_light = df['L'] + df['Solar']
    df['S_L'] = 0.0
    df.loc[(total_light >= L_min) & (total_light <= L_sat), 'S_L'] = 1
    df.loc[total_light > L_sat, 'S_L'] = 0.7

    N_max = 25
    df['P_N'] = df['N'].apply(lambda N: max(0, N - N_max))
    return df


def get_preprocessed_data(file_path):
    df = load_and_clean_data(file_path)
    df = one_hot_encode_weather(df)
    df = calculate_energy_column(df)
    df = calculate_quality_scores(df)
    return df


def get_output_columns(df):
    output_cols = ['E_calculated', 'S_T', 'S_H', 'S_L', 'S_C', 'P_N']
    return df[output_cols]
