import os
import pandas as pd
import matplotlib.pyplot as plt

class LatencyAnalyzer:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.data = []
        self._load_data()

    def _load_data(self):
        for dir_index in range(1, 13):  # Directories from 1.1 to 1.11
            dir_path = os.path.join(self.base_directory, f'2.{dir_index}')
            if os.path.exists(dir_path):
                for file_index in range(10):  # Files from collector_log_0.log to collector_log_7.log
                    file_path = os.path.join(dir_path, f'collector_log_{file_index}.log')
                    if os.path.isfile(file_path):
                        self._parse_log_file(file_path, dir_index, file_index)

    def _parse_log_file(self, file_path, dir_index, file_index):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                timestamp, latency = map(float, line.strip().split(','))
                self.data.append({'timestamp': timestamp, 'latency': latency, 'dir': dir_index, 'file': file_index})

    def generate_plots(self):
        df = pd.DataFrame(self.data)

        # Plot latency over time for each directory
        plt.figure(figsize=(12, 6))
        for dir_index in range(1, 12):
            dir_data = df[df['dir'] == dir_index]
            plt.plot(dir_data['timestamp'], dir_data['latency'], label=f'Directory 1.{dir_index}')
        
        plt.xlabel('Timestamp')
        plt.ylabel('Latency')
        plt.title('Latency over Time for Each Example')
        plt.legend()
        plt.show()

        # Plot overall latency distribution
        plt.figure(figsize=(12, 6))
        df['latency'].hist(bins=100)
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Overall Latency Distribution')
        plt.show()

        # Boxplot for latencies per directory
        plt.figure(figsize=(12, 6))
        df.boxplot(column='latency', by='dir')
        plt.xlabel('Example')
        plt.ylabel('Latency(ms)')
        plt.title('Latency Distribution per Example')
        plt.suptitle('')
        plt.show()

# Usage
base_directory = 'data_dump'  # Replace with the actual base directory path
analyzer = LatencyAnalyzer(base_directory)
analyzer.generate_plots()