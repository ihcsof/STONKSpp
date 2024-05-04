from mosaik.scenario import SimConfig as SimConfig
from mosaik.scenario import World as World
from mosaik._version import __version__ as __version__
import mosaik.util

__all__ = ['World']
# End: Imports

# Sim config
SIM_CONFIG = {
    'Simulator': { 'python': 'SimulatorDiscreteMosaik:Simulator' },
    #'NS3Simulator': {'python': 'ns3_simulator_module.NS3SimulatorClass'}
}
END = 10000 #  10000seconds

# Create World
world = World(SIM_CONFIG)

# Start simulators
sim = world.start('Simulator', eid_prefix='Model_') # ...
# ns3_sim = world.start('NS3Simulator') # ...

# Instantiate models
#model = sim.ExampleModel(boh=42)

# Connect entities
#world.connect(model, monitor, 'val', 'delta')

# Create more entities
more_models = sim.Prosumer.create(-1, init_val=-1)
mosaik.util.connect_many_to_one(world, more_models, 'src', 'dest', 'formatted_msg')
# OR
#world.connect(entities_your_sim, entities_ns3_sim, ('data_to_share',))

# Run the simulation
world.run(until=1000)  # Run simulation for 1000 seconds
