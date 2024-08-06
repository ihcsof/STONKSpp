import os

def process_directory(directory_path):
    numbers = []
    for file_name in sorted(os.listdir(directory_path)):
        if file_name.endswith('.log'):
            with open(os.path.join(directory_path, file_name), 'r') as file:
                for line in file:
                    number = int(line.strip().split(',')[1])
                    numbers.append(number)
    return sorted(numbers)

def concatenate_and_sort_logs(base_directory, num_directories):
    all_numbers = []
    
    for i in range(1, num_directories + 1):
        directory_path = os.path.join(base_directory, f'collectorLogs{i}')
        if os.path.exists(directory_path):
            numbers = process_directory(directory_path)
            all_numbers.extend(numbers)
    
    # Sort the entire list of numbers after concatenation
    all_numbers.sort()

    return all_numbers

def write_output(numbers, output_file):
    with open(output_file, 'w') as file:
        file.write(','.join(map(str, numbers)))

if __name__ == "__main__":
    base_directory = '.'  # Specify the base directory containing the 'collectorLogsX' directories
    num_directories = 2  # Adjust based on the number of directories
    output_file = 'output.txt'  # Specify the output file name

    # Process directories and get sorted numbers
    sorted_numbers = concatenate_and_sort_logs(base_directory, num_directories)

    # Write the sorted numbers to the output file
    write_output(sorted_numbers, output_file)

    print(f"Output written to {output_file}")