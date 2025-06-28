import igraph as ig
import matplotlib.pyplot as plt

def load_graph(file_path):
    """Load the graph from a file."""
    return ig.Graph.Load(file_path, format='picklez')

def remove_self_edges(graph):
    """Remove self-loops (edges where the source and target are the same)."""
    self_edges = [e.index for e in graph.es if e.source == e.target]
    graph.delete_edges(self_edges)

def remove_isolated_vertices(graph):
    """Remove isolated vertices (vertices with no edges)."""
    isolated_vertices = [v.index for v in graph.vs if graph.degree(v) == 0]
    graph.delete_vertices(isolated_vertices)

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
    
    # Remove the axis
    ax.set_axis_off()
    
    # Plot the graph without vertex labels, smaller nodes, and undirected edges
    ig.plot(
        graph,
        layout=layout,
        target=ax,
        vertex_label=None,        # No labels on the nodes
        vertex_size=10,           # Smaller nodes
        vertex_color="lightblue",
        edge_color="gray",
        edge_width=1.0,
        edge_curved=[False] * graph.ecount()        # No curvature for edges (undirected)
    )
    plt.show()

def save_graph(graph, file_path):
    """Save the graph to a file."""
    graph.write_picklez(file_path)

def main():
    input_file = 'graphs/examples/P2P_model_3.pyp2p'
    output_file = 'graphs/examples/biggest5.pyp2p'
    
    # Load the graph
    graph = load_graph(input_file)

    # Remove self-edges
    #remove_self_edges(graph)

    # Remove isolated vertices
    #remove_isolated_vertices(graph)

    # List all edges
    print("All edges in the graph:")
    list_all_edges(graph)

    graph = graph.as_undirected()

    # Print vertex attributes
    for vertex in graph.vs:
        print(f"Vertex {vertex.index}: {vertex.attributes()}")

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

    # Save the graph after cleaning
    save_graph(graph, output_file)
    print(f"Graph saved to {output_file}")

if __name__ == '__main__':
    main()