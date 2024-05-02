import networkx as nx
import matplotlib.pyplot as plt
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
        plt.figure(figsize=(40, 40))
        subgraph = G.subgraph(subgraph_nodes).to_directed()
        
        pos = nx.spring_layout(subgraph)  # Layout for visualization
        nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=10, font_weight='bold', arrows=True, arrowstyle='-|>', arrowsize=20)

        labels = nx.get_edge_attributes(subgraph, 'weight')

        formatted_edge_labels = {edge: f"{labels[edge]:.4f}" for edge in labels}
        # Adding edge labels
        
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=formatted_edge_labels, font_size=20)
    
        plt.savefig('Edge_alpha/graph_'+str(i))   