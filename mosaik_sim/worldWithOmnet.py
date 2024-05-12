from pathlib import Path
import sys
from time import sleep

base_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_path))

from cosima_core.util.util_functions import start_omnet, \
    check_omnet_connection, stop_omnet, \
    log
import cosima_core.util.general_config as cfg
from scenario_config import NETWORK
import mosaik
import mosaik.util

SIMULATION_END = 260000
START_MODE = 'cmd'
NETWORK = 'SimpleNetworkTCP'

# Simulation configuration -> tells mosaik where to find the simulators
SIM_CONFIG = {
    'Simulator': {
        'python': 'SimulatorDiscreteMosaik:Simulator'
    },
    'CommunicationSimulator': {
        'python': 'cosima_core.simulators.communication_simulator:CommunicationSimulator',
    }
}

omnet_process = start_omnet(START_MODE, NETWORK)
check_omnet_connection(cfg.PORT)

# Create mosaik World
world = mosaik.World(SIM_CONFIG, time_resolution=0.001, cache=False)

prosumer_attribute_mapping = {}
for i in range(0, 51):
    prosumer_attribute_mapping[f'Prosumer{i}'] = f'message_with_delay_for_prosumer{i}'

sim = world.start('Simulator', eid_prefix='Model_')

models = sim.Prosumer.create(51, init_val=-1)  # Create 51 instances of Prosumer

comm_sim = world.start('CommunicationSimulator',
                       step_size=1,
                       port=cfg.PORT,
                       client_attribute_mapping=prosumer_attribute_mapping).CommunicationModel()

i = 0
for model in models:
   world.connect(model, comm_sim, f'message')
   world.connect(comm_sim, model, prosumer_attribute_mapping[f'Prosumer{i}'])
   i += 1

# set initial event for simple agent
world.set_initial_event(comm_sim, time=0)

log(f'run until {SIMULATION_END}')
world.run(until=SIMULATION_END)
log("end of process")
sleep(5)
stop_omnet(omnet_process)
