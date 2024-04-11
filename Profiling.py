import cProfile
import pstats
import sys
from Simulator import Simulator

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    sim = Simulator()
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        sim.load_config(config_file)
        sim.Parameters_Test()
    else:
        print("No configuration file provided. Using default parameters.")
    
    sim.StartNewSimulation()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
