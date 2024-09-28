import subprocess
import shutil
import os

# Define the scripts to run
script_cosima_world = 'cosimaWorld.py'
script_cosima_changing = 'cosimaWorldChanging.py'

nruns = 2  # Number of runs for each set

# Define the different run types with their respective arguments
args_2 = ['--network', 'ProsumerSimNetN2', '--size', '10', '--graph', 'Pool_reduced.pyp2p']

run_types = [
    # FIRST GRAPH
    #---------------------------------------------
    #{'name': 'First set', 'script': script_cosima_world, 'args': [], 'name_tag': '1.1'},
    #{'name': 'Second set', 'script': script_cosima_world, 'args': ['--scale-factor', '2'], 'name_tag': '1.2'},
    #{'name': 'Third set', 'script': script_cosima_world, 'args': ['--scale-factor', '3'], 'name_tag': '1.3'},
    #{'name': 'Fourth set', 'script': script_cosima_world, 'args': ['--loss-prob', '0.05', '0.01', '0.01', '0.5', '0.15', '0', '0', '0.35'], 'name_tag': '1.4'},
    #{'name': 'Fifth set', 'script': script_cosima_world, 'args': ['--loss-prob', '0.15'], 'name_tag': '1.5'},
    
    # New sets with --network ProsumerWifiNetwork
    #{'name': 'Network set 1', 'script': script_cosima_world, 'args': ['--network', 'ProsumerWifiNetwork'], 'name_tag': '1.9'},
    #{'name': 'Network set 2', 'script': script_cosima_world, 'args': ['--network', 'ProsumerWifiNetwork', '--scale-factor', '2'], 'name_tag': '1.10'},
    #{'name': 'Network set 3', 'script': script_cosima_world, 'args': ['--network', 'ProsumerWifiNetwork', '--loss-prob', '0.15'], 'name_tag': '1.11'},

     # SECOND GRAPH
    #---------------------------------------------
    #{'name': 'First set', 'script': script_cosima_world, 'args': args_2.copy(), 'name_tag': '2.1'},
    #{'name': 'Second set', 'script': script_cosima_world, 'args': args_2 + ['--scale-factor', '2'], 'name_tag': '2.2'},
    #{'name': 'Third set', 'script': script_cosima_world, 'args': args_2 + ['--scale-factor', '3'], 'name_tag': '2.3'},
    #{'name': 'Fourth set', 'script': script_cosima_world, 'args': args_2 + ['--loss-prob', '0.05', '0.01', '0.01', '0.5', '0.15', '0', '0', '0.35'], 'name_tag': '2.4'},
    #{'name': 'Fifth set', 'script': script_cosima_world, 'args': args_2 + ['--loss-prob', '0.15'], 'name_tag': '2.5'}
]

# Run all sets
for run_type in run_types:
    print(f"Running {run_type['name']} with script {run_type['script']} and name tag {run_type['name_tag']}")
    
    # Create a directory for this run set
    directory = run_type['name_tag']
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f"Created directory: {directory}")
    
    for i in range(1, nruns):
        print(f"Running iteration {i} for {run_type['name']} with name {run_type['name_tag']}")
        
        # Construct the command to run
        command = ['python3', run_type['script'], '--run', str(i), '--name', run_type['name_tag']] + run_type['args']
        
        # Execute the script with the arguments
        subprocess.run(command)
        
        # In the middle of the loop, execute the sudo kill command
        print(f"Attempting to kill process on port 4242 after iteration {i}")
        try:
            process_id = subprocess.check_output(['sudo', 'lsof', '-t', '-i', ':4242']).decode().strip()
            if process_id:
                subprocess.run(['sudo', 'kill', '-9', process_id])
                print(f"Killed process {process_id} on port 4242.")
            else:
                print("No process found on port 4242.")
        except subprocess.CalledProcessError:
            print("No process found on port 4242 or error occurred while checking.")
    
    print(f"Completed all iterations for {run_type['name']}")

# List of scenario configs to substitute
scenario_configs_1 = [
    '/root/models/cosima_core/mosaik_sim/scenario_configs/first.py',
    '/root/models/cosima_core/mosaik_sim/scenario_configs/second.py',
    '/root/models/cosima_core/mosaik_sim/scenario_configs/third.py'
]
scenario_configs_2 = [
    '/root/models/cosima_core/mosaik_sim/scenario_configs/first.py',
    '/root/models/cosima_core/mosaik_sim/scenario_configs/fourth.py',
    '/root/models/cosima_core/mosaik_sim/scenario_configs/fifth.py'
]

def run_scenario_configs(scenario_configs, graph, name_tag_base, extra_args = []):
    # Replace scenario_config.py with each scenario config and run
    for idx, config in enumerate(scenario_configs, start=1):
        config_name = config.split('/')[-1]
        name_tag = f"{graph}.{name_tag_base}"
        print(f"Substituting scenario_config.py with {config_name} and using name tag {name_tag}...")

        # Create a directory for this scenario config run
        directory = name_tag
        if not os.path.exists(directory):
            os.mkdir(directory)
            print(f"Created directory: {directory}")

        try:
            shutil.copy(config, '/root/models/scenario_config.py')
            print(f"Substitution completed with {config_name}.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        # Perform the run with the substituted scenario_config.py
        print(f"Starting runs with substituted scenario_config.py ({config_name}) and name tag {name_tag}...")

        for i in range(1, nruns):
            print(f"Running iteration {i} with substituted scenario_config.py ({config_name}) and name {name_tag}")
            
            # Run cosimaWorldChanging.py for this set
            subprocess.run(['python3', script_cosima_changing, '--run', str(i), '--name', name_tag] + extra_args)
            
            # Kill process on port 4242 if necessary
            print(f"Attempting to kill process on port 4242 after iteration {i}")
            try:
                process_id = subprocess.check_output(['sudo', 'lsof', '-t', '-i', ':4242']).decode().strip()
                if process_id:
                    subprocess.run(['sudo', 'kill', '-9', process_id])
                    print(f"Killed process {process_id} on port 4242.")
                else:
                    print("No process found on port 4242.")
            except subprocess.CalledProcessError:
                print("No process found on port 4242 or error occurred while checking.")

        # Increment the name tag for the next config
        name_tag_base += 1

    print("All iterations completed for all scenario configs.")

#run_scenario_configs(scenario_configs_1, '1', 6)
run_scenario_configs(scenario_configs_2, '2', 6, args_2)