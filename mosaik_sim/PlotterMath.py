import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read the data from the log files
def extract_primal_dual_values(directory):
    primal_values = {}
    dual_values = {}

    # Iterate over each file in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("collector_log_"):
            filepath = os.path.join(directory, filename)
            file_number = int(filename.split('_')[-1])  # Extract the number from the filename
            with open(filepath, 'r') as file:
                primals = []
                duals = []
                for line in file:
                    parts = line.split(',')
                    primal = float(parts[3]) if parts[3] != 'inf' else np.inf
                    dual = float(parts[4]) if parts[4] != 'inf' else np.inf
                    primals.append(primal)
                    duals.append(dual)
                primal_values[file_number] = primals
                dual_values[file_number] = duals

    return primal_values, dual_values

# Function to plot and save the combined average primal and dual values without infinity
def save_combined_average_primal_dual_plot_without_infinity(primal_values, dual_values, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate the average of the primal values (ignoring infinity)
    primal_avg = []
    max_len = max(len(values) for values in primal_values.values())
    
    for i in range(max_len):
        valid_values = [values[i] for values in primal_values.values() if i < len(values) and values[i] != np.inf]
        if valid_values:
            primal_avg.append(np.mean(valid_values))
    
    # Calculate the average of the dual values (ignoring infinity)
    dual_avg = []
    max_len = max(len(values) for values in dual_values.values())
    
    for i in range(max_len):
        valid_values = [values[i] for values in dual_values.values() if i < len(values) and values[i] != np.inf]
        if valid_values:
            dual_avg.append(np.mean(valid_values))
    
    # Plotting the average primal and dual values on the same plot
    plt.figure(figsize=(10, 7))
    plt.plot(primal_avg, label='Average Primal Value', color='blue')
    plt.plot(dual_avg, label='Average Dual Value', color='green')
    plt.title('Average Primal and Dual Values Evolution')
    plt.xlabel('Time(ms)')
    plt.ylabel('Value in log scale')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'avg_prim_dual_val.png'))
    plt.close()

# Assuming the logs are in the 'collectorLogs' directory
directory = 'collectorLogs'
primal_values, dual_values = extract_primal_dual_values(directory)

# Save the combined average plot without infinity values to the current directory
save_combined_average_primal_dual_plot_without_infinity(primal_values, dual_values, output_dir='.')