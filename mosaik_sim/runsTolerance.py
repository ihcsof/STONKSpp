import os
import numpy as np

# List of directories containing the files
directories = ['backups/10mbps', 'backups/10gbps']  # Example directories

# Tolerance for stopping the iteration
tolerance = 0.001  # Example tolerance value

# Function to calculate the cumulative average latency change
def calculate_cumulative_average_latency_change(directory, tolerance):
    # Initialize a list to store cumulative second values
    cumulative_second_values = []

    # Get a sorted list of files to ensure consistent order
    files = sorted(os.listdir(directory))

    # Initialize variable to store the previous cumulative average
    previous_average = None

    # Loop through each file in the directory cumulatively
    for i, filename in enumerate(files):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):  # Check if it's a file
            with open(filepath, 'r') as file:
                for line in file:
                    # Split the line by comma and extract the second value
                    try:
                        value = float(line.split(',')[1])
                        if value < 100:
                            cumulative_second_values.append(value)
                    except (IndexError, ValueError):
                        print(f"Skipping line in {filename}: {line.strip()}")

        # Calculate the current cumulative average latency
        current_average = np.mean(cumulative_second_values)

        # Calculate the change in cumulative average latency if we have a previous value
        if previous_average is not None:
            change = abs(current_average - previous_average)
            print(f"After adding file {i+1} ({filename}): Cumulative Average Latency = {current_average:.4f}, Change = {change:.4f}")

            # Check if the change is within the tolerance
            if change < tolerance:
                print(f"Stopping as change in cumulative average latency is within the tolerance of {tolerance}.")
                break
        else:
            print(f"After adding file {i+1} ({filename}): Cumulative Average Latency = {current_average:.4f}")

        # Update the previous cumulative average
        previous_average = current_average

    return current_average

# Loop through each directory and calculate the cumulative average latency change
for directory in directories:
    print(f"Processing directory: {directory}")
    final_average = calculate_cumulative_average_latency_change(directory, tolerance)
    print(f"Final Cumulative Average Latency for {directory} = {final_average:.4f}\n")