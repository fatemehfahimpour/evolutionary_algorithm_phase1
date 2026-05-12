import pandas as pd
from Preprocessing import get_preprocessed_data, get_output_columns
DATA_PATH = 'data/1_26336110128.csv'

df_clean = get_preprocessed_data(DATA_PATH)

# Get output columns
output = get_output_columns(df_clean)