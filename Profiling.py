import cProfile
import pstats
import sys
from SimulatorDiscreteSync import Simulator

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    sim = Simulator()
    
    sim.load_config("profilingConfig.json")
    sim.Parameters_Test()
    
    sim.StartNewSimulation()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
