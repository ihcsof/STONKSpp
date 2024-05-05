from mosaik.scenario import SimConfig as SimConfig
from mosaik.scenario import World as World
from mosaik._version import __version__ as __version__
import mosaik.util

__all__ = ['World']
# End: Imports

# Sim config
SIM_CONFIG = {
    'Simulator': { 'python': 'SimulatorDiscreteMosaik:Simulator' },
    #'NS3Simulator': {'python': 'ns3_simulator_module:NS3SimulatorClass'}
}
END = 900000

# Create World
world = World(SIM_CONFIG, debug=False)

# Start simulators
sim = world.start('Simulator', eid_prefix='Model_') # ...
# ns3_sim = world.start('NS3Simulator') # ...

# Run the simulation
world.run(until=END)
