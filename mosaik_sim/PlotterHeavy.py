import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

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
        latencies = np.diff(times)  # Calculate the differences between consecutive timesteps
        if len(latencies) > 0:
            mean_latency = np.mean(latencies)  # Calculate the mean latency
        else:
            mean_latency = 0
        mean_latencies[prosumer] = mean_latency
    return mean_latencies

def plot_mean_latencies(mean_latencies, output_path='mean_latencies.png'):
    prosumer_ids = list(mean_latencies.keys())
    latencies = list(mean_latencies.values())

    plt.figure(figsize=(14, 8))
    plt.bar(prosumer_ids, latencies, color='skyblue')
    plt.xlabel('Prosumer ID')
    plt.ylabel('Mean Latency (ms)')
    plt.title('Mean Latency for Each Prosumer')
    plt.xticks(prosumer_ids)
    plt.grid(True)
    plt.savefig(output_path)  # Save the figure to a file
    plt.close()  # Close the figure to free memory

def plot_latency_evolution(data, output_path='latency_evolution.png'):
    plt.figure(figsize=(14, 8))
    for prosumer, times in data.items():
        latencies = np.diff(times)
        steps = range(1, len(latencies) + 1)
        plt.plot(steps, latencies, marker='o', label=f'Prosumer {prosumer}')

    plt.xlabel('Step Number')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Evolution for Each Prosumer Over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)  # Save the figure to a file
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':
    log_directory = 'collectorLogs'
    data = read_log_files(log_directory)
    mean_latencies = calculate_mean_latencies(data)
    plot_mean_latencies(mean_latencies)
    plot_latency_evolution(data)