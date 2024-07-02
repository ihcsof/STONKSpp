import igraph as ig
import random

def load_graph(file_path):
    """Load the graph from a file."""
    return ig.Graph.Load(file_path, format='picklez')

def save_graph(graph, file_path):
    """Save the graph to a file."""
    graph.save(file_path, format='picklez')

def remove_low_degree_nodes(graph, num_nodes_to_select, num_nodes_to_remove):
    """Remove a specified number of nodes with the lowest degree from the graph after shuffling."""
    degrees = graph.degree()
    
    # Get the indices of the nodes sorted by degree in ascending order
    sorted_nodes_by_degree = sorted(range(len(degrees)), key=lambda k: degrees[k])
    
    # Select the top `num_nodes_to_select` nodes with the lowest degree
    nodes_to_select = sorted_nodes_by_degree[:num_nodes_to_select]
    
    # Shuffle the selected nodes
    random.shuffle(nodes_to_select)
    
    # Select the top `num_nodes_to_remove` nodes from the shuffled list
    nodes_to_remove = nodes_to_select[:num_nodes_to_remove]
    
    # Remove the nodes from the graph
    graph.delete_vertices(nodes_to_remove)
    
    return graph

def main():
    input_file = 'graphs/examples/Connected_community_model.pyp2p'
    output_file = 'graphs/examples/Comm_reduced.pyp2p'
    
    # Load the graph
    graph = load_graph(input_file)
    
    num_nodes_to_select = 47
    num_nodes_to_remove = 35
    graph = remove_low_degree_nodes(graph, num_nodes_to_select, num_nodes_to_remove)
    
    # Save the modified graph
    save_graph(graph, output_file)
    print(f"Graph saved to {output_file}")

if __name__ == '__main__':
    main()