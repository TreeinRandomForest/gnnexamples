import networkx as nx
import matplotlib.pyplot as plt
import subprocess

def save_edge_attr_graph(edge_index, edge_attrs, batch_cnt):
    
    G = nx.Graph()
    edge_attrs = edge_attrs.T[0].tolist()
    edge_index = edge_index.T.tolist()
    
    # Add edges with attributes
    for i, edge in enumerate(edge_index):
        G.add_edge(edge[0], edge[1], weight=edge_attrs[i])

    disconnected_subgraphs = list(nx.connected_components(G))
    # Visualization
    i = batch_cnt-1
    for subgraph_nodes in disconnected_subgraphs:
        
        i += 1
        
        subgraph = G.subgraph(subgraph_nodes).to_directed()

        figdim = max(min(len(subgraph.nodes),50),5)
        fig, ax = plt.subplots(figsize=(figdim,figdim))

        pos = nx.spring_layout(subgraph, k=figdim*2)  # Layout for visualization
        nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', node_size=figdim, font_size=figdim/10, font_weight='bold', arrows=True, arrowstyle='-|>', arrowsize=figdim)

        labels = nx.get_edge_attributes(subgraph, 'weight')

        formatted_edge_labels = {edge: f"{labels[edge]:.4f}" for edge in labels}
        # Adding edge labels
        formatted_filtered_edge_labels = {edge: f"{labels[edge]:.4f}" for edge, label in labels.items() if label > 0.00001}
        formatted_filtered_edge_labels_0 = {edge: f"{labels[edge]:.4f}" for edge, label in labels.items() if label <= 0.00001}
        #formatted_filtered_edge_labels = {edge: f"{labels[edge]:.4f}" for edge in formatted_edge_labels}
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=formatted_filtered_edge_labels, font_size=figdim)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=formatted_filtered_edge_labels_0, font_size=figdim/10)
        
        
        plt.savefig('Edge_alpha/graph_'+str(i), dpi=300)
        plt.close()   

def calculate_variable():
    variable = 0.8
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        total_memory_mb = int(output.decode().strip())
        variable = 1.0 - min(1.0, max(0.0, (total_memory_mb / 4096) * 0.10))
    except subprocess.CalledProcessError:
        # Error occurred while running the command
        variable = 0.8
    except ValueError:
        # Error occurred while converting output to integer
        variable = 0.8
    return variable