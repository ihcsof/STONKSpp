from mosaik.scenario import SimConfig as SimConfig
from mosaik.scenario import World as World
from mosaik._version import __version__ as __version__
import mosaik.util

__all__ = ['World']
# End: Imports


# Sim config
SIM_CONFIG = {
    'ExampleSim': {
        'python': 'simulator_mosaik:ExampleSim',
    },
    'Collector': {
        'cmd': '%(python)s Collector.py %(addr)s',
    },
}
END = 10  # 10 seconds
# End: Sim config

# Create World
world = World(SIM_CONFIG)
# End: Create World

# Start simulators
examplesim = world.start('ExampleSim', eid_prefix='Model_')
collector = world.start('Collector')
# End: Start simulators

# Instantiate models
model = examplesim.ExampleModel(init_val=2)
monitor = collector.Monitor()
# End: Instantiate models

# Connect entities
world.connect(model, monitor, 'val', 'delta')
# End: Connect entities

# Create more entities
more_models = examplesim.ExampleModel.create(2, init_val=3)
mosaik.util.connect_many_to_one(world, more_models, monitor, 'val', 'delta')
# End: Create more entities

# Run simulation
world.run(until=END)