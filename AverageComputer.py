import os
import numpy as np

# List of directories containing the files
directories = [
    'backups/2xScaleFactor', 
    'backups/3xScaleFactor', 
    'backups/10gbps', 
    'backups/10mbps', 
    'backups/distributedHeavyTraffic', 
    'backups/distributedTraffic', 
    'backups/mainDisruption', 
    'backups/retransmission', 
    'backups/retransmissionAll', 
    'backups/traffic'
]

# Initialize a dictionary to hold the average latency for each directory
average_latencies = {}

# Loop through each directory
for directory in directories:
    second_values = []
    
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file starts with "collector"
        if not filename.startswith("collector"):
            continue
        
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):  # Check if it's a file
            with open(filepath, 'r') as file:
                for line in file:
                    # Split the line by comma and extract the second value
                    try:
                        value = float(line.split(',')[1])
                        if value < 150:
                            second_values.append(value)
                    except (IndexError, ValueError):
                        print(f"Skipping line in {filename}: {line.strip()}")
    
    # Ensure that second_values is not empty before proceeding
    if len(second_values) == 0:
        raise ValueError(f"No valid second values were found in the files of {directory}.")
    
    # Calculate the average latency for the directory
    average_latency = np.mean(second_values)
    average_latencies[directory] = average_latency

# Display the average latencies for each directory
print("Average Latency for each directory:")
for directory, average_latency in average_latencies.items():
    print(f"{directory}: {average_latency:.2f} ms")