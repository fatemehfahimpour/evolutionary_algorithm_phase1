import pandas as pd
from Preprocessing import get_preprocessed_data, get_output_columns
from genetic_algorithm import run_ga
DATA_PATH = 'data/1_26336110128.csv'

df_clean = get_preprocessed_data(DATA_PATH)
# output = get_output_columns(df_clean)
population , fitness_values = run_ga(df_clean)
print(population)