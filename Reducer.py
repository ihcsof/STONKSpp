import igraph as ig
import random

def load_graph(file_path):
    """Load the graph from a file."""
    return ig.Graph.Load(file_path, format='picklez')

def save_graph(graph, file_path):
    """Save the graph to a file."""
    graph.save(file_path, format='picklez')

def remove_nodes(graph, fraction):
    """Remove a fraction of nodes from the graph."""
    num_nodes = len(graph.vs)
    num_nodes_to_remove = int(num_nodes * fraction)
    
    # Select random nodes to remove
    nodes_to_remove = random.sample(range(num_nodes), num_nodes_to_remove)
    
    # Remove the nodes from the graph
    graph.delete_vertices(nodes_to_remove)
    
    return graph

def main():
    input_file = 'graphs/examples/Pool_model.pyp2p'
    output_file = 'graphs/examples/Pool_reduced.pyp2p'
    
    # Load the graph
    graph = load_graph(input_file)
    
    fraction_to_remove = 0.5
    graph = remove_nodes(graph, fraction_to_remove)
    
    # Save the modified graph
    save_graph(graph, output_file)
    print(f"Graph saved to {output_file}")

if __name__ == '__main__':
    main()