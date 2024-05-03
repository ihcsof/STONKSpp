# demo_1.py
import mosaik
import mosaik.util

# Sim config
SIM_CONFIG: mosaik.SimConfig = {
    'Simulator': {
        'python3': 'SimulatorDiscreteMosaik.Simulator'
    }
}
END = 1000  # 1000 seconds

# Create World
world = mosaik.World(SIM_CONFIG)

# Start simulators
sim = world.start('Simulator', eid_prefix='Model_')

# Instantiate models
#model = sim.ExampleModel(boh=42)

# Connect entities
#world.connect(model, monitor, 'val', 'delta')

# Create more entities
#more_models = examplesim.ExampleModel.create(2, init_val=3)
#mosaik.util.connect_many_to_one(world, more_models, monitor, 'val', 'delta')

world.run(until=END)