from pathlib import Path
import sys
import tempfile

from time import sleep

base_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(base_path))

from cosima_core.util.util_functions import start_omnet, \
    check_omnet_connection, stop_omnet, \
    log
import cosima_core.util.general_config as cfg
CONTENT_PATH = cfg.ROOT_PATH / 'simulators' / 'tic_toc_example' / 'content.csv'

import mosaik
import mosaik.util

SIMULATION_END = 260000
START_MODE = 'cmd'
NETWORK = 'SimpleNetworkTCP'

# Simulation configuration -> tells mosaik where to find the simulators
SIM_CONFIG = {
    'Simulator': {
        'python': 'SimulatorDiscreteMosaik2:Simulator'
    },
    'SimpleAgent': {
        'python': 'simple_agent_simulator:SimpleAgent',
    },
    'CommunicationSimulator': {
        'python': 'cosima_core.simulators.communication_simulator:CommunicationSimulator',
    },
    # 'StatisticsSimulator': {
    #    'python': 'cosima_core.simulators.statistics_simulator:StatisticsSimulator',
    #}
}

omnet_process = start_omnet(START_MODE, NETWORK)
check_omnet_connection(cfg.PORT)

# Create mosaik World
world = mosaik.World(SIM_CONFIG, time_resolution=0.001, cache=False)

client_attribute_mapping = {
    'client0': 'message_with_delay_for_client0',
    'client1': 'message_with_delay_for_client1'
}


prosumer_sim = world.start('Simulator',
                            content_path=CONTENT_PATH,
                            client_name='client0',
                            neighbor='client1').ProsumerSim()
simple_agent = world.start('SimpleAgent',
                             content_path=CONTENT_PATH,
                             client_name='client1',
                             neighbor='client0').SimpleAgentModel()

comm_sim = world.start('CommunicationSimulator',
                       step_size=1,
                       port=cfg.PORT,
                       client_attribute_mapping=client_attribute_mapping).CommunicationModel()

#stat_sim = world.start('StatisticsSimulator', network=NETWORK, save_plots=True).Statistics()  # , step_time=200

world.connect(prosumer_sim, comm_sim, f'message', weak=True)
world.connect(comm_sim, prosumer_sim, client_attribute_mapping['client0'])
world.connect(simple_agent, comm_sim, f'message', weak=True)
world.connect(comm_sim, simple_agent, client_attribute_mapping['client1'])
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
