import os
import numpy as np
import matplotlib.pyplot as plt

# List of directories containing the files
directories = ['backups/10mbps', 'backups/10gbps']  # Example directories

# List of labels corresponding to each directory
labels = ['10 Mbps Network', '10 Gbps Network']  # Example labels

# Initialize a dictionary to hold the second values for each directory
all_second_values = {}

# Loop through each directory
for directory in directories:
    second_values = []
    
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):  # Check if it's a file
            with open(filepath, 'r') as file:
                for line in file:
                    # Split the line by comma and extract the second value
                    try:
                        value = float(line.split(',')[1])
                        if value < 100:
                            second_values.append(value)
                    except (IndexError, ValueError):
                        print(f"Skipping line in {filename}: {line.strip()}")
    
    # Ensure that second_values is not empty before proceeding
    if len(second_values) == 0:
        raise ValueError(f"No valid second values were found in the files of {directory}.")
    
    # Store the second values in the dictionary
    all_second_values[directory] = np.array(second_values)

# Plot the CDF for each directory
plt.figure(figsize=(8, 6))

for directory, label in zip(all_second_values.keys(), labels):
    second_values = all_second_values[directory]
    sorted_values = np.sort(second_values)
    cdf = np.arange(len(sorted_values)) / float(len(sorted_values))
    plt.plot(sorted_values, cdf, linestyle='-', label=f'CDF for {label}')

plt.xlabel('Latency (ms)')
plt.ylabel('Probability')
plt.title('Latency CDF for perfect network')
plt.grid(True)
plt.legend()
cdf_filepath = 'collectorLogs/combined_cdf.png'
plt.savefig(cdf_filepath)
plt.show()

# Create a combined boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(all_second_values.values(), vert=True, labels=labels)
plt.ylabel('Latency (ms)')
plt.title('Latency Boxplot for perfect network')
plt.grid(True)
boxplot_filepath = 'collectorLogs/combined_boxplot.png'
plt.savefig(boxplot_filepath)
plt.show()

print(f"Combined CDF plot saved to {cdf_filepath}")
print(f"Combined Boxplot saved to {boxplot_filepath}")