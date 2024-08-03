import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
        latencies = np.diff(times)
        if len(latencies) > 0:
            mean_latency = np.mean(latencies)
        else:
            mean_latency = 0
        mean_latencies[prosumer] = mean_latency
    return mean_latencies

def calculate_average_of_averages(directories):
    averages = []
    for log_directory in directories:
        data = read_log_files(log_directory)
        mean_latencies = calculate_mean_latencies(data)
        avg_of_avg = np.mean(list(mean_latencies.values()))
        averages.append(avg_of_avg)
    return averages

def plot_average_of_averages(averages, output_path='data_dump/average_of_averages.png'):
    plt.figure(figsize=(14, 8))
    dir_labels = [f'F{i+1}' for i in range(len(averages))]

    plt.bar(dir_labels, averages, color='skyblue')
    plt.xlabel('Log Directories')
    plt.ylabel('Average of Mean Latencies (ms)')
    plt.title('Average of Mean Latencies for Multiple Directories')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    directories = ['data_dump/1.2', 'data_dump/1.3', 'data_dump/1.4', 'data_dump/1.5']
    averages = calculate_average_of_averages(directories)
    plot_average_of_averages(averages)