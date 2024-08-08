import subprocess

script_to_run = 'cosimaWorldChanging.py'

for i in range(1, 3):
    print(f"Running iteration {i}")
    
    subprocess.run(['python3', script_to_run, "--run", str(i)])
    
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
        # This will catch the error if lsof returns a non-zero exit status
        print("No process found on port 4242 or error occurred while checking.")
    
print("All iterations completed.")