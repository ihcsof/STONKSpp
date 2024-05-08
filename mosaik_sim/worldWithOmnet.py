from pathlib import Path
import sys
import mosaik

base_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_path))

print("Python path after insertion:", sys.path)

from cosima_core.util.util_functions import start_omnet, check_omnet_connection, stop_omnet, log
import cosima_core.util.general_config as cfg

# Define the simulation end time and other constants
SIMULATION_END = 260000
START_MODE = 'cmd'
NETWORK = 'SimpleNetworkTCP'
ONTENT_PATH = cfg.ROOT_PATH / 'simulators' / 'tic_toc_example' / 'content.csv'

# Simulation configuration
SIM_CONFIG = {
    'Simulator': {
        'python': 'SimulatorDiscreteMosaik:Simulator'
    },
    'OmnetSimulator': {
        'python': 'cosima_core.simulators.omnetpp_connection:OmnetppConnection'
    }
}

# Start the OMNeT++ process and check the connection
omnet_process = start_omnet(START_MODE, NETWORK)
check_omnet_connection(cfg.PORT)

# Create the Mosaik world with the specified simulation configuration
world = mosaik.World(SIM_CONFIG, time_resolution=0.001, cache=False)

# Start the custom Simulator and the OMNeT++ Simulator
sim = world.start('Simulator', eid_prefix='Model_')
omnet_sim = world.start('OmnetSimulator', step_size=1, port=cfg.PORT, network=NETWORK).OmnetModel()

# Instantiate multiple models within your simulator
models = sim.Prosumer.create(51, init_val=-1)  # Create 51 instances of Prosumer

# Connect each Prosumer to the OMNeT++ simulator for data exchange
for model in models:
    world.connect(model, omnet_sim, ('data_to_omnet',))
    world.connect(omnet_sim, model, ('data_from_omnet',))

# Run the simulation until the defined end time
log(f'Running the simulation until {SIMULATION_END}')
world.run(until=SIMULATION_END)
log("End of simulation process")

# Clean up after simulation
sleep(5)
stop_omnet(omnet_process)