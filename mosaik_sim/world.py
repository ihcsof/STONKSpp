# demo_1.py
import mosaik
import mosaik.util

# Sim config
SIM_CONFIG: mosaik.SimConfig = {
    'Simulator': { 'python3': 'SimulatorDiscreteMosaik.Simulator' },
    #'NS3Simulator': {'python': 'ns3_simulator_module.NS3SimulatorClass'}
}
END = 1000  # 1000 seconds

# Create World
world = mosaik.World(SIM_CONFIG)

# Start simulators
sim = world.start('Simulator', eid_prefix='Model_') # ...
# ns3_sim = world.start('NS3Simulator') # ...

# Instantiate models
#model = sim.ExampleModel(boh=42)

# Connect entities
#world.connect(model, monitor, 'val', 'delta')

# Create more entities
#more_models = examplesim.ExampleModel.create(2, init_val=3)
#mosaik.util.connect_many_to_one(world, more_models, monitor, 'val', 'delta')
# OR
#world.connect(entities_your_sim, entities_ns3_sim, ('data_to_share',))

# Run the simulation
world.run(until=1000)  # Run simulation for 1000 seconds
