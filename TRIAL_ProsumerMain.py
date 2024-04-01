# Import the Prosumer class from Prosumer.py
from Prosumer import Prosumer
import numpy as np

def main():
    # Initialize a Prosumer instance with updated mock data including 'costfct'
    agent_info = {
        'Type': 'Consumer',  # or 'Manager' based on your setup
        'AssetsNum': 1,
        'Assets': [{ 'Pmin': 0, 'Pmax': 10, 'costfct': 'Quadratic', 'costfct_coeff': [1, 2], 'p_bounds_up': 10,'p_bounds_low': 0}]
    }
    prosumer = Prosumer(agent=agent_info)

    # Number of iterations to simulate
    simulation_steps = 10

    # Simulation loop
    for step in range(simulation_steps):
        # Simulate behavior or interaction
        # Example: Randomly adjust the Pmax of the first asset
        new_Pmax = np.random.uniform(5, 15)  # Random new Pmax within a range
        prosumer.data.Pmax[0] = new_Pmax

        # Output: Print current state
        print(f"Step {step}:")
        print(f"  Type: {prosumer.data.type}")
        print(f"  Asset 0 Pmax: {prosumer.data.Pmax[0]}")
        # This is where you could add any other simulation logic or interactions

if __name__ == "__main__":
    main()