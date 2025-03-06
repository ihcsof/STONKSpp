import igraph as ig
import matplotlib.pyplot as plt
import random

def load_graph(file_path):
    """Load the graph from a file."""
    return ig.Graph.Load(file_path, format='picklez')

def list_all_edges(graph):
    edges = graph.get_edgelist()
    for edge in edges:
        print(f"Vertex {edge[0]} is connected to Vertex {edge[1]}")
    print(f"\nTotal edges: {len(edges)}")

def list_neighbors(graph, vertex_id):
    neighbors = graph.neighbors(vertex_id)
    print(f"Vertex {vertex_id} is connected to: {neighbors}")

def adjacency_list(graph):
    for vertex in range(graph.vcount()):
        neighbors = graph.neighbors(vertex)
        print(f"Vertex {vertex} is connected to: {neighbors}")

def plot_graph(graph):
    layout = graph.layout("kk")  # Kamada-Kawai layout
    fig, ax = plt.subplots()
    ig.plot(
        graph,
        layout=layout,
        target=ax,
        vertex_label=range(graph.vcount()),
        vertex_size=20,
        vertex_color="lightblue",
        edge_color="gray",
        edge_width=1.0,
        edge_curved=[False] * graph.ecount()
    )
    plt.show()

def add_random_nodes(graph, num_new_nodes):
    """Add random nodes and edges to the graph, ensuring each new node has 'ID', 'Partners', and 'Assets' attributes."""
    for _ in range(num_new_nodes):
        # Add a new vertex (node)
        graph.add_vertices(1)
        
        new_node = graph.vcount() - 1  # The most recently added node
        
        # Assign a unique ID for the new node
        graph.vs[new_node]['ID'] = new_node
        
        # Initialize the 'Partners' attribute as an empty list
        graph.vs[new_node]['Partners'] = []
        
        # Initialize the 'Assets' attribute as an empty list
        graph.vs[new_node]['Assets'] = []
        
        # Randomly select an existing node to connect to
        existing_node = random.randint(0, graph.vcount() - num_new_nodes - 1)
        
        # Ensure no self-edges are created
        if new_node != existing_node:
            graph.add_edges([(new_node, existing_node)])
            edge = graph.es.find(_source=new_node, _target=existing_node)
            
            # Assign a default weight to the edge
            edge['weight'] = 1.0
            
            # Add the new connection to the 'Partners' attribute
            graph.vs[new_node]['Partners'].append(graph.vs[existing_node]['ID'])
            graph.vs[existing_node]['Partners'].append(graph.vs[new_node]['ID'])
    
    print(f"\n{num_new_nodes} new nodes added to the graph.")
    print(f"Total edges after adding nodes: {graph.ecount()}")

def remove_self_edges(graph):
    """Remove all self-edges from the graph."""
    # Find all edges where the source and target vertex are the same
    self_edges = graph.es.select(lambda e: e.source == e.target)
    
    # Delete the self-edges
    graph.delete_edges(self_edges)
    print(f"\nRemoved {len(self_edges)} self-edges from the graph.")
    print(f"Total edges after removing self-edges: {graph.ecount()}")

def save_graph(graph, file_path):
    """Save the graph to a file."""
    graph.save(file_path, format='picklez')
    print(f"\nGraph saved as {file_path}")

def main():
    input_file = 'graphs/examples/P2P_model_reduced.pyp2p'
    output_file = 'graphs/examples/enlarged.pyp2p'
    
    # Load the graph
    graph = load_graph(input_file)

    # List all edges
    print("All edges in the graph:")
    list_all_edges(graph)
    
    # List neighbors for a specific vertex
    vertex_id = 0  # Replace with the desired vertex ID
    print(f"\nNeighbors of vertex {vertex_id}:")
    list_neighbors(graph, vertex_id)
    
    # Print the adjacency list
    print("\nAdjacency list of the graph:")
    adjacency_list(graph)
    
    # Add random nodes
    num_new_nodes = 5  # Specify how many new nodes you want to add
    add_random_nodes(graph, num_new_nodes)
    
    # Remove self-edges
    remove_self_edges(graph)

    # Plot the graph with the new nodes
    print("\nPlotting the graph with new nodes:")
    plot_graph(graph)

    # Save the modified graph
    save_graph(graph, output_file)

if __name__ == '__main__':
    main()
