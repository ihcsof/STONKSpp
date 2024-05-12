#from mosaik.scenario import SimConfig as SimConfig
#from mosaik.scenario import World as World
#from mosaik._version import __version__ as __version__
import mosaik
import mosaik.util

__all__ = ['World']

# Sim config
SIM_CONFIG = {
    'Simulator': { 'python': 'SimulatorDiscreteMosaik:Simulator' },
    'Collector': {
        'cmd': '%(python)s Collector.py %(addr)s',
    },
    #'NS3Simulator': {'python': 'ns3_simulator_module.NS3SimulatorClass'}
}
END = 260000

# Create World
world = mosaik.World(SIM_CONFIG, debug=False)

# Start simulators
sim = world.start('Simulator', eid_prefix='Model_') # ...
collector = world.start('Collector')
# ns3_sim = world.start('NS3Simulator') # ...

# Instantiate models
monitor = collector.Monitor()
#model = sim.Prosumer(init_val=2)
#world.connect(model, monitor, 'src', 'dest', 'formatted_msg')

# Create more entities
models = sim.Prosumer.create(51, init_val=-1) # TEMP (FIXED PARAMS)
mosaik.util.connect_many_to_one(world, models, monitor, 'prosumer_msg')
# OR
#world.connect(entities_your_sim, entities_ns3_sim, ('data_to_share',))

# Run the simulation
world.run(until=END)