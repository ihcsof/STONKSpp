import igraph as ig
import random

def generate_new_vertex(index):
    """Generate attributes for a new vertex."""
    asset_types = ['Flats', 'Solar', 'Stores', 'Factory', 'Fossil', 'Appliances', 'Wind', 'Load']
    vertex_type = 'Agent' if index % 10 != 0 else 'Manager'  # Create managers at intervals
    assets_num = random.randint(1, 2) if vertex_type == 'Agent' else 0
    assets = []

    if vertex_type == 'Agent':
        for i in range(assets_num):
            asset_type = random.choice(asset_types)
            # Cost function coefficients closer to the original example
            costfct_coeff = [round(random.uniform(0.0001, 0.001), 6), round(random.uniform(0.16, 0.21), 6), 0]
            # Power bounds similar to the original example
            p_bounds_up = round(random.uniform(-50, 100), 4)
            p_bounds_low = round(random.uniform(-200, -50), 4)
            assets.append({
                'name': asset_type,
                'id': i + 1,
                'type': asset_type,
                'costfct': 'Quadratic',
                'costfct_coeff': costfct_coeff,
                'longitude': 0,
                'latitude': 0,
                'bus': 0,
                'p_bounds_up': p_bounds_up,
                'p_bounds_low': p_bounds_low,
                'q_bounds_up': 0,
                'q_bounds_low': 0
            })

    # Ensure at least one partner
    partners = random.sample(range(index), k=random.randint(1, 3)) if index > 0 else []

    return {
        'ID': index,
        'Type': vertex_type,
        'AssetsNum': assets_num,
        'Assets': assets,
        'Partners': partners,  # Ensure non-zero partners
        'Preferences': [0],
        'Community': [random.choice([62, 64, 65])],  # Assign to a community
        'CommPref': [0],
        'CommGoal': 'Lowest Price' if vertex_type == 'Manager' else None,
        'ImpFee': 0,
        'name': f"Build {index}" if vertex_type == 'Agent' else f"Community {index}"
    }

def enlarge_graph(graph, target_size=200):
    """Enlarge the graph with new vertices and edges."""
    current_size = graph.vcount()

    # Add new vertices
    for i in range(current_size, target_size):
        new_vertex_attrs = generate_new_vertex(i)
        graph.add_vertex(**new_vertex_attrs)

    # Add new edges and ensure each node has at least one edge
    for i in range(current_size, target_size):
        # Ensure every new vertex has at least one edge
        v1 = random.randint(0, i-1) if i > 0 else 0
        # Assign edge weights that are closer to the original example
        weight = round(random.uniform(0.5, 3.0), 2)  # Adjusting weights to be more reasonable
        graph.add_edge(i, v1, weight=weight)

    # Add additional random edges to maintain coherence and add weights
    for _ in range(target_size * 3):  # Arbitrary factor for edge density
        v1 = random.randint(0, target_size - 1)
        v2 = random.randint(0, target_size - 1)
        if v1 != v2 and not graph.are_connected(v1, v2):
            weight = round(random.uniform(0.5, 3.0), 2)  # Assign weights in a reasonable range
            graph.add_edge(v1, v2, weight=weight)

    return graph

def main():
    input_file = 'graphs/examples/new_graph2.pyp2p'
    
    # Load the existing graph
    graph = ig.Graph.Load(input_file, format='picklez')

    # Enlarge the graph
    enlarged_graph = enlarge_graph(graph, target_size=200)

    # Save the enlarged graph
    output_file = 'graphs/examples/enlarged_graph.pyp2p'
    enlarged_graph.save(output_file, format='picklez')
    
    return output_file

if __name__ == '__main__':
    main()
import igraph as ig
import random

def generate_new_vertex(index):
    """Generate attributes for a new vertex."""
    asset_types = ['Flats', 'Solar', 'Stores', 'Factory', 'Fossil', 'Appliances', 'Wind', 'Load']
    vertex_type = 'Agent' if index % 10 != 0 else 'Manager'  # Create managers at intervals
    assets_num = random.randint(1, 2) if vertex_type == 'Agent' else 0
    assets = []

    if vertex_type == 'Agent':
        for i in range(assets_num):
            asset_type = random.choice(asset_types)
            # Cost function coefficients closer to the original example
            costfct_coeff = [round(random.uniform(0.0001, 0.001), 6), round(random.uniform(0.16, 0.21), 6), 0]
            # Power bounds similar to the original example
            p_bounds_up = round(random.uniform(-50, 100), 4)
            p_bounds_low = round(random.uniform(-200, -50), 4)
            assets.append({
                'name': asset_type,
                'id': i + 1,
                'type': asset_type,
                'costfct': 'Quadratic',
                'costfct_coeff': costfct_coeff,
                'longitude': 0,
                'latitude': 0,
                'bus': 0,
                'p_bounds_up': p_bounds_up,
                'p_bounds_low': p_bounds_low,
                'q_bounds_up': 0,
                'q_bounds_low': 0
            })

    # Ensure at least one partner
    partners = random.sample(range(index), k=random.randint(1, 3)) if index > 0 else []

    return {
        'ID': index,
        'Type': vertex_type,
        'AssetsNum': assets_num,
        'Assets': assets,
        'Partners': partners,  # Ensure non-zero partners
        'Preferences': [0],
        'Community': [random.choice([62, 64, 65])],  # Assign to a community
        'CommPref': [0],
        'CommGoal': 'Lowest Price' if vertex_type == 'Manager' else None,
        'ImpFee': 0,
        'name': f"Build {index}" if vertex_type == 'Agent' else f"Community {index}"
    }

def enlarge_graph(graph, target_size=200):
    """Enlarge the graph with new vertices and edges."""
    current_size = graph.vcount()

    # Add new vertices
    for i in range(current_size, target_size):
        new_vertex_attrs = generate_new_vertex(i)
        graph.add_vertex(**new_vertex_attrs)

    # Add new edges and ensure each node has at least one edge
    for i in range(current_size, target_size):
        # Ensure every new vertex has at least one edge
        v1 = random.randint(0, i-1) if i > 0 else 0
        # Assign edge weights that are closer to the original example
        weight = round(random.uniform(0.5, 3.0), 2)  # Adjusting weights to be more reasonable
        graph.add_edge(i, v1, weight=weight)

    # Add additional random edges to maintain coherence and add weights
    for _ in range(target_size * 3):  # Arbitrary factor for edge density
        v1 = random.randint(0, target_size - 1)
        v2 = random.randint(0, target_size - 1)
        if v1 != v2 and not graph.are_connected(v1, v2):
            weight = round(random.uniform(0.5, 3.0), 2)  # Assign weights in a reasonable range
            graph.add_edge(v1, v2, weight=weight)

    return graph

def main():
    input_file = 'graphs/examples/new_graph2.pyp2p'
    
    # Load the existing graph
    graph = ig.Graph.Load(input_file, format='picklez')

    # Enlarge the graph
    enlarged_graph = enlarge_graph(graph, target_size=200)

    # Save the enlarged graph
    output_file = 'graphs/examples/enlarged_graph.pyp2p'
    enlarged_graph.save(output_file, format='picklez')
    
    return output_file

if __name__ == '__main__':
    main()