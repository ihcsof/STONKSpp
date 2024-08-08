import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the files
directory = 'collectorLogs'

# Initialize a list to hold the second values
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
                    second_values.append(value)
                except (IndexError, ValueError):
                    print(f"Skipping line in {filename}: {line.strip()}")

# Convert the list to a numpy array for further processing
second_values = np.array(second_values)

# Ensure that second_values is not empty before proceeding
if len(second_values) == 0:
    raise ValueError("No valid second values were found in the files.")

# Sort the data for CDF
sorted_values = np.sort(second_values)
cdf = np.arange(len(sorted_values)) / float(len(sorted_values))

# Plot the CDF with an interpolated line
plt.figure(figsize=(8, 6))
plt.plot(sorted_values, cdf, linestyle='-', marker='')  # Use a line without markers
plt.xlabel('Latency (ms)')
plt.ylabel('Probability')
plt.title('Latency CDF for perfect network')
plt.grid(True)
cdf_filepath = os.path.join(directory, 'cdf.png')
plt.savefig(cdf_filepath)
plt.show()

# Create a boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(second_values, vert=False)
plt.xlabel('Latency (ms)')
plt.title('Latency Boxplot for perfect network')
plt.grid(True)
boxplot_filepath = os.path.join(directory, 'boxplot.png')
plt.savefig(boxplot_filepath)
plt.show()

print(f"CDF plot saved to {cdf_filepath}")
print(f"Boxplot saved to {boxplot_filepath}")