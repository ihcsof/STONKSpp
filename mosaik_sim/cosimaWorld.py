# alias sium='sudo kill -9 $(sudo lsof -t -i :4242)'

from pathlib import Path
import sys
import argparse

from time import sleep
import cProfile
import pstats
import random

base_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_path))

from cosima_core.util.util_functions import start_omnet, \
    check_omnet_connection, stop_omnet, \
    log
import cosima_core.util.general_config as cfg

import mosaik
import mosaik.util

parser = argparse.ArgumentParser(description='Run simulation with specified prosumer step size.')
parser.add_argument('--name', type=str, default='collectorLogs', help='Name of the log file')
parser.add_argument('--step-size', type=int, default=1, help='Step size for the prosumer simulator')
parser.add_argument('--scale-factor', type=int, default=1, help='Scale factor for optimization')
parser.add_argument('--loss-prob', nargs='+', default=[], type=float, help="List of loss probabilities")
parser.add_argument('--network', type=str, default='ProsumerSimNetN', help="Network to simulate")
parser.add_argument('--size', type=int, default=8, help="Size of the network")
parser.add_argument('--graph', type=str, default='P2P_model_reduced.pyp2p', help="Graph name")

# for multiple run purposes (index of the run loop: see runs.py)
parser.add_argument('--run', type=int, default=1, help='Loop iteration index (default: 0)')

args = parser.parse_args()

SIMULATION_END = 999999999
START_MODE = 'cmd'
NETWORK = args.network
NUM_PROSUMERS = args.size

# Simulation configuration -> tells mosaik where to find the simulators
SIM_CONFIG = {
    'Simulator': {
        #'python': 'SimulatorDiscreteCosima:Simulator'
        'python': 'SDCWithCalcLatencies:Simulator'
    },
    'Collector': {
        'python': 'Collector:Collector',
    },
    'CommunicationSimulator': {
        'python': 'cosima_core.simulators.communication_simulator:CommunicationSimulator',
    },
}

omnet_process = start_omnet(START_MODE, NETWORK)
check_omnet_connection(cfg.PORT)

# Create mosaik World
world = mosaik.World(SIM_CONFIG, time_resolution=1, cache=True)

client_attribute_mapping = {}
for i in range(0, NUM_PROSUMERS + 1):
    client_attribute_mapping[f'client{i}'] = f'message_with_delay_for_client{i}'

prosumer_sim = world.start('Simulator',
                            client_name=f'client{NUM_PROSUMERS}',
                            name=args.name,
                            graph=args.graph,
                            step_size=args.step_size,
                            scale_factor=args.scale_factor,
                            run=args.run).ProsumerSim()

comm_sim = world.start('CommunicationSimulator',
                       step_size=0.001,
                       port=cfg.PORT,
                       client_attribute_mapping=client_attribute_mapping).CommunicationModel()

world.connect(prosumer_sim, comm_sim, f'message', weak=True)
world.connect(comm_sim, prosumer_sim, client_attribute_mapping[f'client{NUM_PROSUMERS}'])

collectors = [None] * NUM_PROSUMERS
for i in range(0, NUM_PROSUMERS):
    collectors[i] = world.start('Collector', 
                                client_name=f'client{i}', 
                                simulator=f'client{NUM_PROSUMERS}', 
                                loss_prob=args.loss_prob,
                                run=args.run).Collector()

    world.connect(collectors[i], comm_sim, f'message', weak=True)
    world.connect(comm_sim, collectors[i], client_attribute_mapping[f'client{i}'])

profiler = cProfile.Profile()
profiler.enable()

# set initial event for simple agent
world.set_initial_event(prosumer_sim.sid, time=0)

log(f'run until {SIMULATION_END}')
print(args)
world.run(until=SIMULATION_END)
#world.run(until=SIMULATION_END, rt_factor=1/100)
log("end of process")
sleep(5)
stop_omnet(omnet_process)

profiler.disable()

# Generate a random integer and append to the filename
'''random_number = random.randint(1000, 9999)
filename = f'profiling_output_{random_number}.txt'
with open(filename, 'w') as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats('cumulative')
    stats.print_stats()

print(f"Profiling results are written to '{filename}'.")'''
