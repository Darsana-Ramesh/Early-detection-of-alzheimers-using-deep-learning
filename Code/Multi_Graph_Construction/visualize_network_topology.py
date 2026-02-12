import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# ================= CONFIGURATION =================
BASE_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction"
DIRS = {
    "Pearson (Static)": os.path.join(BASE_DIR, "Graph_Output"),
    "Coherence (Freq)": os.path.join(BASE_DIR, "Graph_Output_Coherence"),
    "Granger (Directed)": os.path.join(BASE_DIR, "Graph_Output_Granger")
}

# Thresholds to keep the graph readable (0.0 to 1.0)
# Only edges stronger than this will be drawn
THRESHOLDS = {
    "Pearson (Static)": 0.6,   # Strong correlations only
    "Coherence (Freq)": 0.7,   # High coherence only
    "Granger (Directed)": 0.2  # Granger values are usually smaller
}
# =================================================

def plot_graph(matrix, title, threshold, is_directed=False):
    # 1. Create Graph Object
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # 2. Add 116 Nodes
    n_nodes = matrix.shape[0]
    for i in range(n_nodes):
        G.add_node(i)
    
    # 3. Add Edges (Only if above threshold)
    rows, cols = np.where(np.abs(matrix) > threshold)
    for r, c in zip(rows, cols):
        if r != c: # Skip self-loops
            weight = matrix[r, c]
            G.add_edge(r, c, weight=weight)
            
    # 4. Draw
    pos = nx.circular_layout(G) # Circular layout is best for brain connectivity
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue')
    
    # Draw Edges (Color by weight)
    if is_directed:
        nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Reds, 
                               width=1.0, arrowstyle='->', arrowsize=10)
    else:
        nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Blues, 
                               width=1.0)
        
    plt.title(f"{title}\n(Threshold > {threshold})")
    plt.axis('off')

# --- MAIN EXECUTION ---
print("--- Generating Network Topology Plots ---")

# Find a subject that exists in all 3 folders
# We'll just grab the first file from Pearson and try to find matches
sample_file = list(Path(DIRS["Pearson (Static)"]).glob("*.npy"))[0]
subject_id = sample_file.name.split('_corr_matrix')[0]

print(f"Visualizing Subject: {subject_id}")

plt.figure(figsize=(18, 6))

# Plot 1: Pearson
plt.subplot(1, 3, 1)
path = os.path.join(DIRS["Pearson (Static)"], f"{subject_id}_corr_matrix.npy")
if os.path.exists(path):
    mat = np.load(path)
    # Zero out negative correlations for simpler visualization if desired
    mat[mat < 0] = 0 
    plot_graph(mat, "Pearson (Functional)", THRESHOLDS["Pearson (Static)"], is_directed=False)

# Plot 2: Coherence
plt.subplot(1, 3, 2)
path = os.path.join(DIRS["Coherence (Freq)"], f"{subject_id}_coherence_matrix.npy")
if os.path.exists(path):
    mat = np.load(path)
    plot_graph(mat, "Coherence (Frequency)", THRESHOLDS["Coherence (Freq)"], is_directed=False)

# Plot 3: Granger
plt.subplot(1, 3, 3)
path = os.path.join(DIRS["Granger (Directed)"], f"{subject_id}_granger_matrix.npy")
if os.path.exists(path):
    mat = np.load(path)
    plot_graph(mat, "Granger (Effective/Directed)", THRESHOLDS["Granger (Directed)"], is_directed=True)

plt.tight_layout()
print("Opening visualization window...")
plt.show()