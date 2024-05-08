import cProfile
import pstats
import sys
import random
import matplotlib.pyplot as plt
from SimulatorDiscrete import Simulator

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    sim = Simulator()
    
    sim.load_config("profilingConfig.json")
    sim.Parameters_Test()
    
    sim.run()

    profiler.disable()

    plt.hist(sim.latency_times, bins=20, color='skyblue', edgecolor='black')
    plt.title('Latency Distribution')
    plt.xlabel('Latency Time (s)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Generate a random integer and append to the filename
    random_number = random.randint(1000, 9999)
    filename = f'profiling_output_{random_number}.txt'
    with open(filename, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()
    
    print(f"Profiling results are written to '{filename}'.")