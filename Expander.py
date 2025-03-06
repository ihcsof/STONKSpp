import pandas as pd
import numpy as np

def perturb_values(val, percent=10):
    return val * (1 + np.random.randn() * percent / 100)

def expand_data(file_path, output_path, expansion_factor=90):
    data = pd.read_csv(file_path)
    expanded_data = pd.DataFrame()
    
    for _ in range(expansion_factor):
        temp_data = data.copy()
        for col in ['Pmin', 'Pmax', 'a', 'b']:
            temp_data[col] = temp_data[col].apply(perturb_values)
        expanded_data = pd.concat([expanded_data, temp_data], ignore_index=True)
    
    expanded_data.to_csv(output_path, index=False)
    print(f"Data expanded and saved to {output_path}")

file_path = 'graphs/toexpand.csv'
output_path = 'graphs/toserialize.csv'
expand_data(file_path, output_path, expansion_factor=50) 