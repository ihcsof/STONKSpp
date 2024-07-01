import igraph as ig
import matplotlib.pyplot as plt

def load_graph(file_path):
    """Load the graph from a file."""
    return ig.Graph.Load(file_path, format='picklez')

def list_all_edges(graph):
    edges = graph.get_edgelist()
    for edge in edges:
        print(f"Vertex {edge[0]} is connected to Vertex {edge[1]}")

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

def main():
    input_file = 'graphs/examples/P2P_model_reduced.pyp2p'
    
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
    
    # Plot the graph
    print("\nPlotting the graph:")
    plot_graph(graph)

if __name__ == '__main__':
    main()