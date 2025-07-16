import igraph as ig

# =========================================
# Simple Community Detection Script
# =========================================

def load_graph(file_path):
    """Load the graph from a file."""
    return ig.Graph.Load(file_path, format='picklez')


def detect_communities(graph, method='louvain'):
    """
    Detect communities and return the clustering object.
    Supported methods: 'louvain', 'fastgreedy', 'spinglass'
    """
    if method == 'louvain':
        return graph.community_multilevel()
    elif method == 'fastgreedy':
        dendro = graph.community_fastgreedy()
        return dendro.as_clustering()
    elif method == 'spinglass':
        return graph.community_spinglass()
    else:
        raise ValueError(f"Unsupported method '{method}'")


def print_communities(clustering):
    """Print each community as a list of vertex indices."""
    communities = clustering.as_cover() if hasattr(clustering, 'as_cover') else clustering
    for idx, comm in enumerate(communities):
        print(f"Community {idx}: {sorted(comm)}")


def main():
    input_file = 'graphs/examples/P2P_model_4.pyp2p'

    # Load and simplify graph
    g = load_graph(input_file)
    ug = g.as_undirected(mode='collapse')
    ug.simplify(multiple=True, loops=True)

    # Detect communities
    clustering = detect_communities(ug, method='louvain')

    # Print communities
    print_communities(clustering)

if __name__ == '__main__':
    main()
