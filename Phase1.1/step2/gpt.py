"""
stage2_ga.py
نفر دوم: طراحی کروموزوم و الگوریتم ژنتیک برای مرحله دوم
- ورودی: ضرایب بهینه a_norm و b_norm از مرحله اول
- خروجی: بهترین سناریوی ورودی (T_in, T_out, H_in, L, N, CO2, Weather) به همراه انرژی و آسایش
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# وارد کردن توابع کمکی از مرحله اول (برای محاسبه انرژی و امتیازات کیفیت)
# مسیردهی مناسب را بر اساس ساختار پروژه خود تنظیم کنید
from step1.Preprocessing import get_k_w  # تابع استخراج ضریب آب و هوا
# توابع محاسبه امتیازات (S_T, S_H, S_L, S_C, P_N) را مستقیماً از Preprocessing استفاده می‌کنیم
# یا در صورت نیاز آن‌ها را تکرار می‌کنیم تا وابستگی کاهش یابد.
# برای سادگی، آن‌ها را دوباره در اینجا تعریف می‌کنیم (می‌توانید از import استفاده کنید)

# ========================== 1. قیود فیزیکی (بر اساس مستند) ==========================
BOUNDS = {
    'T_in': (10.0, 35.0),
    'T_out': (-5.0, 45.0),
    'H_in': (10.0, 90.0),
    'L': (0.0, 1000.0),
    'N': (1, 30),          # صحیح
    'CO2': (300.0, 1700.0),
}
WEATHER_LIST = ["night", "sunny", "cloudy", "humid", "rainy", "stormy", "cold"]

# ضرایب بهینه از مرحله اول (مقادیر نمونه – آن‌ها را با خروجی GA مرحله اول جایگزین کنید)
# فرض می‌شود پس از اجرای مرحله اول، این مقادیر به دست آمده‌اند.
BEST_A_NORM = [0.17, 0.08, 0.28, 0.07, 0.30, 0.11]   # a_norm (6 تایی)
BEST_B_NORM = [0.18, 0.31, 0.16, 0.27, 0.08]        # b_norm (5 تایی)

# ========================== 2. توابع محاسبه انرژی و آسایش ==========================
def calculate_energy(T_in, T_out, H_in, L, N, CO2, weather_onehot):
    """محاسبه مصرف انرژی طبق فرمول تحلیلی مستند (مرحله دوم)"""
    delta_T = abs(T_in - T_out)
    P_hum = max(0, H_in - 60)**2 + max(0, 30 - H_in)**2
    # محاسبه k_w با استفاده از تابع get_k_w از Preprocessing
    # برای این کار باید یک دیکشنری موقت بسازیم
    weather_dict = {f'W_{w}': 1 if weather_onehot[i]==1 else 0 for i, w in enumerate(WEATHER_LIST)}
    # اضافه کردن کلیدهای دیگر (مقادیر 0) برای جلوگیری از خطا در get_k_w
    for w in WEATHER_LIST:
        if f'W_{w}' not in weather_dict:
            weather_dict[f'W_{w}'] = 0
    # فراخوانی تابع get_k_w (از Preprocessing)
    # اگر نمی‌توانید import کنید، آن را در اینجا کپی کنید
    k_w = get_k_w(pd.Series(weather_dict)) if 'pd' in dir() else 1.0
    # در صورت عدم دسترسی، از یک دیکشنری ساده استفاده می‌کنیم:
    k_w_map = {"night":0.8, "sunny":0.9, "cloudy":1.0, "humid":1.08,
               "rainy":1.15, "stormy":1.25, "cold":1.3}
    for i, w in enumerate(WEATHER_LIST):
        if weather_onehot[i] == 1:
            k_w = k_w_map[w]
            break
    energy = k_w * (0.8 * delta_T**2 + 12 * np.log(1 + N) + 0.02 * P_hum)
    energy += 6 * np.log(1 + L) + 2 * np.log(1 + CO2)
    return energy

def comfort_score(T_in, H_in, L, CO2, N, Solar=0):
    """محاسبه امتیاز آسایش (f2_score بدون نرمال‌سازی اضافی)"""
    # امتیاز دما
    if 20 <= T_in <= 24:
        ST = 1.0
    elif (18 <= T_in < 20) or (24 < T_in <= 26):
        ST = 0.6
    elif (16 <= T_in < 18) or (26 < T_in <= 28):
        ST = 0.2
    else:
        ST = 0.0
    # امتیاز رطوبت
    if 45 <= H_in <= 60:
        SH = 1.0
    elif (35 <= H_in < 45) or (60 < H_in <= 70):
        SH = 0.5
    else:
        SH = 0.0
    # امتیاز نور
    total_light = L + Solar
    if total_light < 30:
        SL = 0.0
    elif total_light <= 900:
        SL = 1.0
    else:
        SL = 0.7
    # امتیاز CO2
    if 800 <= CO2 <= 1200:
        SC = 1.0
    elif (700 <= CO2 < 800) or (1200 < CO2 <= 1400):
        SC = 0.4
    else:
        SC = 0.0
    # جریمه تراکم (N_max=25)
    PN = max(0, N - 25)
    # محاسبه f2_raw با ضرب ضرایب b_norm
    f2_raw = (BEST_B_NORM[0]*ST + BEST_B_NORM[1]*SH + BEST_B_NORM[2]*SL +
              BEST_B_NORM[3]*SC - BEST_B_NORM[4]*PN)
    # نرمال‌سازی ساده به [0,1] (بر اساس بازه‌های تجربی یا کلیپ)
    # در مستند از نرمال‌سازی با مین و مکس دیتاست استفاده می‌شد؛ در اینجا فرض می‌کنیم f2_raw در [-1,1] است
    # برای سادگی، مقدار را در [0,1] محدود می‌کنیم
    return np.clip(f2_raw, 0.0, 1.0)

# ========================== 3. ساختار کروموزوم و جمعیت ==========================
def generate_individual():
    """تولید یک کروموزوم تصادفی با رعایت محدودیت‌ها"""
    T_in = random.uniform(*BOUNDS['T_in'])
    T_out = random.uniform(*BOUNDS['T_out'])
    H_in = random.uniform(*BOUNDS['H_in'])
    L = random.uniform(*BOUNDS['L'])
    N = random.randint(*BOUNDS['N'])
    CO2 = random.uniform(*BOUNDS['CO2'])
    # آب و هوا (one-hot)
    weather_vector = [0]*len(WEATHER_LIST)
    weather_vector[random.randint(0, len(WEATHER_LIST)-1)] = 1
    return [T_in, T_out, H_in, L, N, CO2] + weather_vector

def initialize_population(pop_size):
    return [generate_individual() for _ in range(pop_size)]

# ========================== 4. توابع انتخاب (همانند مرحله اول) ==========================
def roulette_selection(population, fitness_vals):
    fitness = np.array(fitness_vals, dtype=float)
    fitness = np.maximum(fitness, 0)
    total = np.sum(fitness)
    if total == 0:
        return random.choice(population).copy()
    probs = fitness / total
    idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()

def tournament_selection(population, fitness_vals, k=3):
    indices = np.random.choice(len(population), size=k, replace=False)
    best_idx = max(indices, key=lambda i: fitness_vals[i])
    return population[best_idx].copy()

# ========================== 5. تقاطع (نوع‑آگاه) ==========================
def arithmetic_crossover_stage2(p1, p2, alpha=None):
    """تقاطع حسابی با مدیریت انواع داده (حقیقی، صحیح، باینری)"""
    if alpha is None:
        alpha = random.random()
    # بخش 6 ژن اول (حقیقی)
    child1 = [0]*len(p1)
    child2 = [0]*len(p2)
    for i in range(6):  # T_in, T_out, H_in, L, CO2 (N را جداگانه)
        c1 = alpha * p1[i] + (1 - alpha) * p2[i]
        c2 = alpha * p2[i] + (1 - alpha) * p1[i]
        # اعمال کران
        key = ['T_in','T_out','H_in','L','CO2'][i] if i<5 else 'CO2'  # i=4: N? نه، N در اندیس 4 است اما حقیقی نیست
        # برای i=0..4 به ترتیب: T_in, T_out, H_in, L, ? (N در اندیس 4 است اما صحیح)
        # ساده‌تر: برای هر ژن کران مخصوصش را اعمال کنیم
        if i == 0: low, high = BOUNDS['T_in']
        elif i == 1: low, high = BOUNDS['T_out']
        elif i == 2: low, high = BOUNDS['H_in']
        elif i == 3: low, high = BOUNDS['L']
        elif i == 4: low, high = BOUNDS['N']  # اما N صحیح است
        elif i == 5: low, high = BOUNDS['CO2']
        child1[i] = np.clip(c1, low, high)
        child2[i] = np.clip(c2, low, high)
    # ژن N (اندیس 4) - صحیح
    n1 = p1[4]; n2 = p2[4]
    child1[4] = int(round(alpha * n1 + (1-alpha) * n2))
    child2[4] = int(round(alpha * n2 + (1-alpha) * n1))
    child1[4] = np.clip(child1[4], *BOUNDS['N'])
    child2[4] = np.clip(child2[4], *BOUNDS['N'])
    # بخش آب و هوا (باینری) - 7 ژن آخر
    w1 = p1[6:]; w2 = p2[6:]
    # تقاطع یکنواخت برای هر بیت با احتمال 0.5
    child_w1 = [w1[i] if random.random()<0.5 else w2[i] for i in range(len(w1))]
    child_w2 = [w2[i] if random.random()<0.5 else w1[i] for i in range(len(w1))]
    # اطمینان از یکتا بودن بیت 1
    for cw in [child_w1, child_w2]:
        if sum(cw) == 0:
            cw[random.randint(0, len(cw)-1)] = 1
        elif sum(cw) > 1:
            idx = cw.index(1)
            cw[:] = [1 if i==idx else 0 for i in range(len(cw))]
    child1 = child1[:6] + child_w1
    child2 = child2[:6] + child_w2
    return child1, child2

def uniform_crossover_stage2(p1, p2, prob=0.5):
    """تقاطع یکنواخت برای هر ژن به طور مستقل"""
    child1 = []
    child2 = []
    for i in range(len(p1)):
        if random.random() < prob:
            child1.append(p1[i])
            child2.append(p2[i])
        else:
            child1.append(p2[i])
            child2.append(p1[i])
    # اعمال محدودیت‌ها
    for i in range(6):
        if i == 0: low, high = BOUNDS['T_in']
        elif i == 1: low, high = BOUNDS['T_out']
        elif i == 2: low, high = BOUNDS['H_in']
        elif i == 3: low, high = BOUNDS['L']
        elif i == 4: low, high = BOUNDS['N']
        elif i == 5: low, high = BOUNDS['CO2']
        child1[i] = np.clip(child1[i], low, high)
        child2[i] = np.clip(child2[i], low, high)
    child1[4] = int(round(child1[4]))
    child2[4] = int(round(child2[4]))
    # تصحیح one-hot
    for child in [child1, child2]:
        w = child[6:]
        if sum(w) == 0:
            w[random.randint(0, len(w)-1)] = 1
        elif sum(w) > 1:
            idx = w.index(1)
            w[:] = [1 if i==idx else 0 for i in range(len(w))]
        child[6:] = w
    return child1, child2

# ========================== 6. جهش (نوع‑آگاه) ==========================
def mutate_individual(individual, mutation_rate=0.05, sigma=0.5):
    mutated = individual.copy()
    # جهش برای 6 ژن اول (حقیقی و صحیح)
    for i in range(6):
        if random.random() < mutation_rate:
            if i == 4:  # N صحیح
                delta = random.choice([-1, 0, 1])
                mutated[i] += delta
                mutated[i] = int(np.clip(mutated[i], *BOUNDS['N']))
            else:  # حقیقی
                delta = np.random.normal(0, sigma)
                mutated[i] += delta
                if i == 0: low, high = BOUNDS['T_in']
                elif i == 1: low, high = BOUNDS['T_out']
                elif i == 2: low, high = BOUNDS['H_in']
                elif i == 3: low, high = BOUNDS['L']
                elif i == 5: low, high = BOUNDS['CO2']
                mutated[i] = np.clip(mutated[i], low, high)
    # جهش برای آب و هوا (با احتمال mutation_rate، کل بردار عوض شود)
    if random.random() < mutation_rate:
        new_idx = random.randint(0, len(WEATHER_LIST)-1)
        mutated[6:] = [0]*len(WEATHER_LIST)
        mutated[6+new_idx] = 1
    return mutated

# ========================== 7. تابع برازندگی ==========================
def fitness_stage2(individual, lambda_real=0.0, real_penalty_func=None):
    """
    محاسبه fitness برای مرحله دوم:
    F = w1*(1 - E_norm) + w2*comfort_norm - lambda_real * P_real
    با w1=0.4, w2=0.6 (طبق مستند)
    انرژی با نرمال‌سازی خطی ساده (E_min~0, E_max~2000) به [0,1] تبدیل می‌شود.
    """
    T_in, T_out, H_in, L, N, CO2 = individual[:6]
    weather = individual[6:]
    energy = calculate_energy(T_in, T_out, H_in, L, N, CO2, weather)
    comfort = comfort_score(T_in, H_in, L, CO2, N, Solar=0)
    # نرمال‌سازی انرژی (مقادیر تقریبی)
    E_min, E_max = 0.0, 2000.0
    E_norm = (energy - E_min) / (E_max - E_min + 1e-8)
    E_norm = np.clip(E_norm, 0.0, 1.0)
    w1, w2 = 0.4, 0.6
    fitness = w1 * (1 - E_norm) + w2 * comfort
    if lambda_real > 0 and real_penalty_func is not None:
        fitness -= lambda_real * real_penalty_func(individual)
    return fitness

# ========================== 8. الگوریتم ژنتیک اصلی ==========================
def run_ga_stage2(pop_size=80, generations=150,
                  selection_method='tournament', crossover_method='arithmetic',
                  crossover_rate=0.9, mutation_rate=0.05, tournament_size=3,
                  lambda_real=0.0, real_penalty=None, plot_convergence=True):
    """اجرای الگوریتم ژنتیک برای مرحله دوم"""
    population = initialize_population(pop_size)
    fitness_vals = [fitness_stage2(ind, lambda_real, real_penalty) for ind in population]
    best_fitness_history = []
    avg_fitness_history = []
    for gen in range(generations):
        best_fitness_history.append(max(fitness_vals))
        avg_fitness_history.append(np.mean(fitness_vals))
        new_pop = []
        new_fit = []
        while len(new_pop) < pop_size:
            # انتخاب
            if selection_method == 'tournament':
                p1 = tournament_selection(population, fitness_vals, tournament_size)
                p2 = tournament_selection(population, fitness_vals, tournament_size)
            else:  # roulette
                p1 = roulette_selection(population, fitness_vals)
                p2 = roulette_selection(population, fitness_vals)
            # تقاطع
            if random.random() < crossover_rate:
                if crossover_method == 'arithmetic':
                    c1, c2 = arithmetic_crossover_stage2(p1, p2)
                else:  # uniform
                    c1, c2 = uniform_crossover_stage2(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            # جهش
            c1 = mutate_individual(c1, mutation_rate)
            c2 = mutate_individual(c2, mutation_rate)
            # ارزیابی
            fit1 = fitness_stage2(c1, lambda_real, real_penalty)
            fit2 = fitness_stage2(c2, lambda_real, real_penalty)
            new_pop.append(c1)
            new_fit.append(fit1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
                new_fit.append(fit2)
        population = new_pop
        fitness_vals = new_fit
    best_idx = np.argmax(fitness_vals)
    best_individual = population[best_idx]
    best_fitness = fitness_vals[best_idx]
    if plot_convergence:
        plt.figure(figsize=(12,5))
        plt.plot(best_fitness_history, label='Best Fitness')
        plt.plot(avg_fitness_history, label='Avg Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Convergence - Stage2 GA')
        plt.legend()
        plt.grid()
        plt.show()
    return best_individual, best_fitness, best_fitness_history, avg_fitness_history

# ========================== 9. اجرای آزمایشی و خروجی ==========================
if __name__ == '__main__':
    # مثال اجرا با مقادیر پیش‌فرض
    best, fit, hist_best, hist_avg = run_ga_stage2(
        pop_size=80,
        generations=150,
        selection_method='tournament',
        crossover_method='arithmetic',
        mutation_rate=0.05,
        plot_convergence=True
    )
    print("\n=== بهترین سناریوی ورودی (مرحله دوم) ===")
    print(f"T_in = {best[0]:.2f} °C")
    print(f"T_out = {best[1]:.2f} °C")
    print(f"H_in = {best[2]:.2f} %")
    print(f"L = {best[3]:.2f} lux")
    print(f"N = {int(best[4])} عدد گیاه")
    print(f"CO2 = {best[5]:.2f} ppm")
    weather_idx = best[6:].index(1)
    print(f"آب و هوا = {WEATHER_LIST[weather_idx]}")
    energy_val = calculate_energy(*best[:6], best[6:])
    comfort_val = comfort_score(best[0], best[2], best[3], best[5], int(best[4]), Solar=0)
    print(f"\nمصرف انرژی محاسبه‌شده: {energy_val:.2f}")
    print(f"امتیاز آسایش (f2_score نرمال‌شده): {comfort_val:.4f}")
    print(f"Fitness نهایی: {fit:.6f}")