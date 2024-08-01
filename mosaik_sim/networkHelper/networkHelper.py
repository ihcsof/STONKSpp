import json

def generate_network_config(config_file):
    # Read the JSON config file
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    # Extract values from the config
    name = config.get('name', 'MyNet')
    device_name = config.get('deviceName', 'mainSwitch')
    traffic_device = config.get('trafficDevice', 'yes')
    n_prosumers = config.get('nProsumers', 10)
    cable_len = config.get('cableLen', 100)
    comm_type = config.get('commType', 'MyEth10M')  # Default value is 'MyEth10M'
    router_comm_type = config.get('routerCommType', 'Eth10M')  # Default value is 'Eth10M'
    router_cable_len = config.get('routerCableLen', '1000')  # Default value is '1000km'

    # Calculate the number of clients (nProsumers + 1)
    n_clients = n_prosumers + 1

    # Create the network configuration string
    network_config = f"""
network {name}
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