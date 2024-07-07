import os
import networkx as nx
import glob
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Path to the directory containing message log files
log_files_path = 'messagesLogs/message_log_*.log'

# Read each log file to add directed edges
for log_file in glob.glob(log_files_path):
    # Extract the destination prosumer number from the filename
    destination_prosumer = os.path.basename(log_file).split('_')[-1].split('.')[0]
    
    # Open and read the log file
    with open(log_file, 'r') as file:
        for line in file:
            source_prosumer, n_messages_received = line.strip().split(',')
            source_prosumer = source_prosumer.strip()
            n_messages_received = int(n_messages_received.strip())
            
            # Add a directed edge from the source to the destination prosumer
            G.add_edge(source_prosumer, destination_prosumer, weight=n_messages_received)

# To visualize the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G)  # positions for all nodes
weights = nx.get_edge_attributes(G, 'weight')

# Draw nodes and edges
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold', arrows=True, connectionstyle='arc3,rad=0.1')

# Draw edge labels with offset to avoid overlap
edge_labels = nx.get_edge_attributes(G, 'weight')

for (u, v), label in edge_labels.items():
    # Calculate positions for label offset
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    # Offset the labels slightly for better visibility
    if (u, v) in G.edges and (v, u) in G.edges:
        xm_offset = (y2 - y1) * 0.1
        ym_offset = (x1 - x2) * 0.1
        plt.text(xm + xm_offset, ym + ym_offset, label, fontsize=10, color='red', horizontalalignment='center')
        reverse_label = G[v][u]['weight']
        plt.text(xm - xm_offset, ym - ym_offset, reverse_label, fontsize=10, color='blue', horizontalalignment='center')
    else:
        plt.text(xm, ym, label, fontsize=10, color='red', horizontalalignment='center')

# Save the graph to a PNG file
plt.savefig("prosumer_graph.png")