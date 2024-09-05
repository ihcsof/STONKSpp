import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

# Function to read data from a file
def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into components
            components = line.strip().split(',')
            # Parse components into the appropriate data types
            client_id = components[0].split(':')[1]
            latency = float(components[1])
            primal = float(components[3])
            dual = float(components[4])
            data.append((client_id, latency, primal, dual))
    return data

# Input the filename
filename = "backups/10mbps/collector_log_1"

# Read data from the file
data = read_data_from_file(filename)

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['id', 'latency', 'primal', 'dual'])

# Remove rows with inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['primal', 'dual'], inplace=True)

# Remove duplicates based on 'id' and keep the first occurrence
df_unique = df.drop_duplicates(subset=['id'])

# Calculate cumulative latencies
df_unique['cumulative_latency'] = df_unique['latency'].cumsum()

# Interpolate the data for smooth lines
x_new = np.linspace(df_unique['cumulative_latency'].min(), df_unique['cumulative_latency'].max(), 300)

# Interpolate for primal
primal_spline = make_interp_spline(df_unique['cumulative_latency'], df_unique['primal'], k=3)
primal_smooth = primal_spline(x_new)

# Interpolate for dual
dual_spline = make_interp_spline(df_unique['cumulative_latency'], df_unique['dual'], k=3)
dual_smooth = dual_spline(x_new)

# Generate the plot
plt.figure(figsize=(10, 6))
plt.plot(x_new, primal_smooth, label='Primal')
plt.plot(x_new, dual_smooth, label='Dual')

plt.xlabel('Time(ms)')
plt.ylabel('Values')
plt.yscale('log')
plt.title('Primal and Dual Values evolution (in log scale)')
plt.legend()
plt.grid(True)

# Save the plot to a file
output_folder = "backups"
output_filename = "primal_dual_plot.png"
plt.savefig(f"{output_folder}{output_filename}")

plt.close()  # Close the plot to free memory
