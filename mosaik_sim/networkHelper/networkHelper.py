import json
import random
import itertools

def generate_network_config(config_file):
    # Read the JSON config file
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    # Extract values from the config
    name = config.get('name', 'MyNet')
    device_name = config.get('deviceName', 'mainSwitch')
    traffic_device = config.get('trafficDevice', 'yes')
    n_prosumers = config.get('nProsumers', 10)
    cable_len = config.get('cableLen', random.randint(1, 1000))
    comm_type = config.get('commType', 'MyEth10M')
    router_comm_type = config.get('routerCommType', 'Eth10M')
    router_cable_len = config.get('routerCableLen', random.randint(1, 1000))
    randomizer = config.get('randomizer', 'no')
    redundancy = min(max(config.get('redundancy', 0), 0), 100)  # Ensure redundancy is between 0 and 100

    # Calculate the number of clients (nProsumers + 1)
    n_clients = n_prosumers + 1

    # Create the network configuration string
    network_config = f"""network {name}
{{
    @display("bgb=3000,3000");
    submodules:
        scenarioManager: MosaikScenarioManager {{
            @display("p=100,2850");
        }}

        schedulerModule: MosaikSchedulerModule {{
            @display("p=200,2850");
        }}

        configurator: Ipv4NetworkConfigurator {{
            @display("p=300,2850");
        }}

        visualizer: IntegratedCanvasVisualizer {{
            @display("p=400,2850");
        }}

        router: Router {{
            @display("p=577.2,638.3");
        }}

        {device_name}: EtherSwitch {{
            @display("p=1500,1500");
        }}
    """

    # Add traffic device submodule if the option is "yes"
    if traffic_device.lower() == "yes":
        network_config += f"""
        traffic_device_0: StandardHost {{
            @display("p=714.89,206.998");
        }}"""

    # Add client submodules
    for i in range(n_clients):
        network_config += f"""
        client{i}: StandardHost {{ @display("p={200 + i * 200},100"); }}"""

    # Randomizer option: add random routers and switches
    random_connections = ""
    random_devices = []
    if randomizer.lower() == 'yes':
        n_random_routers = random.randint(1, 5)
        n_random_switches = random.randint(1, 5)

        # Add random routers
        for i in range(n_random_routers):
            network_config += f"""
        router{i+1}: Router {{ @display("p={random.randint(500, 1500)},{random.randint(500, 1500)}"); }}"""
            random_devices.append(f'router{i+1}')

        # Add random switches
        for i in range(n_random_switches):
            network_config += f"""
        switch{i+1}: EtherSwitch {{ @display("p={random.randint(500, 1500)},{random.randint(500, 1500)}"); }}"""
            random_devices.append(f'switch{i+1}')

        # Generate random connections between the random devices
        for device1 in random_devices:
            for device2 in random_devices:
                if device1 != device2:
                    random_connections += f"""
        {device1}.ethg++ <--> {router_comm_type} {{ length = {random.randint(50, 1000)}km; }} <--> {device2}.ethg++;"""

        # Connect some existing clients to random devices
        for i in range(n_clients):
            if random.choice([True, False]):  # Randomly decide if a client gets an additional connection
                selected_device = random.choice(random_devices)
                random_connections += f"""
        client{i}.ethg++ <--> {router_comm_type} {{ length = {random.randint(50, 1000)}km; }} <--> {selected_device}.ethg++;"""

        # Add a connection between mainSwitch and a new router or between mainRouter and a new switch
        if random.choice([True, False]):
            selected_router = random.choice([d for d in random_devices if d.startswith('router')])
            random_connections += f"""
        {device_name}.ethg++ <--> {router_comm_type} {{ length = {random.randint(50, 1000)}km; }} <--> {selected_router}.ethg++;"""
        else:
            selected_switch = random.choice([d for d in random_devices if d.startswith('switch')])
            random_connections += f"""
        router.ethg++ <--> {router_comm_type} {{ length = {random.randint(50, 1000)}km; }} <--> {selected_switch}.ethg++;"""

    # Redundancy option: add direct connections between prosumers
    if redundancy > 0:
        # Calculate the number of redundant connections to add
        max_connections = n_clients * (n_clients - 1) // 2  # Maximum possible unique connections between clients
        n_redundant_connections = max_connections * redundancy // 100

        # Generate all possible pairs of clients
        client_pairs = list(itertools.combinations(range(n_clients), 2))
        random.shuffle(client_pairs)  # Shuffle to ensure randomness

        # Add the calculated number of redundant connections
        for pair in client_pairs[:n_redundant_connections]:
            client1, client2 = pair
            random_connections += f"""
        client{client1}.ethg++ <--> {comm_type} {{ length = {random.randint(50, 1000)}m; }} <--> client{client2}.ethg++;"""

    network_config += "\n\n    connections:"

    # Add connection from device_name to router
    network_config += f"""
        {device_name}.ethg++ <--> {router_comm_type} {{ length = {router_cable_len}km; }} <--> router.ethg++;
    """

    # Add connection for the traffic device if the option is "yes"
    if traffic_device.lower() == "yes":
        network_config += f"""
        traffic_device_0.ethg++ <--> {comm_type} {{ length = {cable_len}m; }} <--> {device_name}.ethg++;
    """

    # Add connections for each client
    for i in range(n_clients):
        network_config += f"""
        client{i}.ethg++ <--> {comm_type} {{ length = {cable_len}m; }} <--> {device_name}.ethg++;
    """

    # Append the random connections generated earlier
    network_config += random_connections

    network_config += "\n}"

    # Save the network configuration to a file
    output_file = f"{name}_network.ned"
    with open(output_file, 'w') as file:
        file.write(network_config)

    print(f"Network configuration file '{output_file}' generated successfully.")

if __name__ == "__main__":
    # Example usage
    config_file = 'config.json'
    generate_network_config(config_file)