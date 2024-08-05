import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_log_files(log_directory):
    data = {}
    for filename in os.listdir(log_directory):
        if filename.startswith('collector_log_') and filename.endswith('.log'):
            client_name = filename[len('collector_log_'):-len('.log')]
            with open(os.path.join(log_directory, filename), 'r') as f:
                for line in f:
                    src, time = line.strip().split(',')
                    if src not in data:
                        data[src] = []
                    data[src].append(int(time))
    return data

def calculate_mean_latencies(data):
    mean_latencies = {}
    for prosumer, times in data.items():
        latencies = np.diff(times)
        # Handle zero latencies by replacing them with the last non-zero latency
        latencies_fixed = []
        last_valid_latency = None
        for latency in latencies:
            if latency == 0 and last_valid_latency is not None:
                latencies_fixed.append(last_valid_latency)
            else:
                latencies_fixed.append(latency)
                if latency > 0:
                    last_valid_latency = latency

        if len(latencies_fixed) > 0:
            mean_latency = np.mean(latencies_fixed)  # Calculate the mean latency
        else:
            mean_latency = 0
        mean_latencies[prosumer] = mean_latency
    return mean_latencies

def plot_latency_evolution(data, output_path='data_dump/latency_evolution.png', num_points=15):
    plt.figure(figsize=(14, 8))
    
    # Determine the global maximum time across all prosumers
    global_max_time = max(max(times) for times in data.values())

    for prosumer in sorted(data.keys()):  # Sort prosumer IDs
        times = data[prosumer]
        latencies = np.diff(times)

        # Handle zero latencies by replacing them with the last non-zero latency
        latencies_fixed = []
        last_valid_latency = None
        for latency in latencies:
            if latency == 0 and last_valid_latency is not None:
                latencies_fixed.append(last_valid_latency)
            else:
                latencies_fixed.append(latency)
                if latency > 0:
                    last_valid_latency = latency

        # Use the actual time values for the x-axis
        steps = times[1:]  # Use time values starting from the second entry

        # Remove duplicate time values to avoid interpolation issues
        unique_steps, indices = np.unique(steps, return_index=True)
        unique_latencies = np.array(latencies_fixed)[indices]

        non_zero_indices = np.where(unique_latencies != 0)
        unique_steps = unique_steps[non_zero_indices]
        unique_latencies = unique_latencies[non_zero_indices]

        # Interpolating to reduce the number of points
        if len(unique_steps) > num_points:
            steps_new = np.linspace(min(unique_steps), max(unique_steps), num_points)
            f_interp = interp1d(unique_steps, unique_latencies, kind='linear', bounds_error=False, fill_value="extrapolate")
            latencies_new = f_interp(steps_new)
        else:
            steps_new = unique_steps
            latencies_new = unique_latencies

        plt.plot(steps_new, latencies_new, marker='o', label=f'Prosumer {prosumer}')

    plt.xlabel('Steps')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Evolution for Each Prosumer Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)  # Save the figure to a file
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    log_directory = 'collectorLogs'
    data = read_log_files(log_directory)
    mean_latencies = calculate_mean_latencies(data)
    #plot_mean_latencies(mean_latencies)
    plot_latency_evolution(data)