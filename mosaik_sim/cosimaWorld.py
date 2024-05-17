# alias sium='sudo kill -9 $(sudo lsof -t -i :4242)'

from pathlib import Path
import sys
import argparse

from time import sleep

base_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_path))

from cosima_core.util.util_functions import start_omnet, \
    check_omnet_connection, stop_omnet, \
    log
import cosima_core.util.general_config as cfg

import mosaik
import mosaik.util

SIMULATION_END = 3000
START_MODE = 'cmd'
NETWORK = 'ProsumerSimNet'

# Simulation configuration -> tells mosaik where to find the simulators
SIM_CONFIG = {
    'Simulator': {
        'python': 'SimulatorDiscreteCosima:Simulator'
    },
    'Collector': {
        'python': 'Collector:Collector',
    },
    'CommunicationSimulator': {
        'python': 'cosima_core.simulators.communication_simulator:CommunicationSimulator',
    },
    # 'StatisticsSimulator': {
    #    'python': 'cosima_core.simulators.statistics_simulator:StatisticsSimulator',
    #}
}

parser = argparse.ArgumentParser(description='Run simulation with specified prosumer step size.')
parser.add_argument('--step-size', type=int, default=5000, help='Step size for the prosumer simulator')

args = parser.parse_args()

omnet_process = start_omnet(START_MODE, NETWORK)
check_omnet_connection(cfg.PORT)

# Create mosaik World
world = mosaik.World(SIM_CONFIG, time_resolution=0.001, cache=False)

client_attribute_mapping = {
    'client0': 'message_with_delay_for_client0',
    'client1': 'message_with_delay_for_client1'
}

prosumer_sim = world.start('Simulator',
                            client_name='client0',
                            collector='client1',
                            step_size=args.step_size).ProsumerSim()
collector = world.start('Collector',
                             client_name='client1',
                             simulator='client0').Collector()

comm_sim = world.start('CommunicationSimulator',
                       step_size=1,
                       port=cfg.PORT,
                       client_attribute_mapping=client_attribute_mapping).CommunicationModel()

#stat_sim = world.start('StatisticsSimulator', network=NETWORK, save_plots=True).Statistics()  # , step_time=200

world.connect(prosumer_sim, comm_sim, f'message', weak=True)
world.connect(comm_sim, prosumer_sim, client_attribute_mapping['client0'])

world.connect(collector, comm_sim, f'message', weak=True)
world.connect(comm_sim, collector, client_attribute_mapping['client1'])
#world.connect(prosumer_sim, stat_sim, 'message', time_shifted=True, initial_data={'message': None})
#world.connect(stat_sim, prosumer_sim, 'stats')
#world.connect(stat_sim, simple_agent, 'stats')

# set initial event for simple agent
world.set_initial_event(prosumer_sim.sid, time=0)

log(f'run until {SIMULATION_END}')
world.run(until=SIMULATION_END)
log("end of process")
sleep(5)
stop_omnet(omnet_process)
