import matplotlib.pyplot as plt
import numpy as np

def read_numbers_from_file(input_file):
    with open(input_file, 'r') as file:
        data = file.read()
    # Split the data by commas and convert to integers
    numbers = list(map(int, data.split(',')))
    return numbers

def calculate_latencies(numbers):
    latencies = [numbers[0]]  # Start with the first value as the initial latency
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1]:
            # If the numbers are the same, use the previous latency
            latency = latencies[-1]
        else:
            # Otherwise, compute the difference as the latency
            latency = numbers[i] - numbers[i - 1]
        latencies.append(latency)
    return latencies

def plot_cdf(latencies, output_file_cdf):
    sorted_data = np.sort(latencies)
    y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    plt.figure(figsize=(8, 6))
    plt.step(sorted_data, y_values, where='post')  # 'post' ensures the stairs effect
    plt.title('CDF Plot of Latencies')
    plt.xlabel('Latency')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.savefig(output_file_cdf)
    plt.show()

def plot_boxplot(latencies, output_file_boxplot):
    plt.figure(figsize=(8, 6))
    plt.boxplot(latencies, vert=False)
    plt.title('Boxplot of Latencies')
    plt.xlabel('Latency')
    plt.grid(True)
    plt.savefig(output_file_boxplot)
    plt.show()

if __name__ == "__main__":
    input_file = 'output.txt'  # File produced by the previous script
    output_file_cdf = 'cdf_plot.png'
    output_file_boxplot = 'boxplot.png'

    # Read numbers from file
    numbers = read_numbers_from_file(input_file)

    # Calculate latencies
    latencies = calculate_latencies(numbers)

    # Generate CDF Plot
    plot_cdf(latencies, output_file_cdf)

    # Generate Boxplot
    plot_boxplot(latencies, output_file_boxplot)

    print(f"CDF plot saved as {output_file_cdf}")
    print(f"Boxplot saved as {output_file_boxplot}")