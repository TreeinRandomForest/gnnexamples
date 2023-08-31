import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import MeanAggregation

# Define the GNN architecture
class GNNModel(nn.Module):
    def __init__(self,
                 in_node_dim=2,
                 out_dim=20,
                 layer_dims=[16, 32]):

        super().__init__()

        layer_dims = [in_node_dim] + layer_dims

        self.conv_layers = nn.ModuleList({})

        for idx in range(1, len(layer_dims)):

            layer = GCNConv(in_channels=layer_dims[idx-1],
                            out_channels=layer_dims[idx])
            
            self.conv_layers.append(layer)

        self.fc = nn.Linear(layer_dims[-1], out_dim)

        self.aggr = MeanAggregation()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for l in self.conv_layers:
            x = l(x, edge_index)
            x = x.relu()

        x = self.fc(x)

        if 'ptr' in data:
            x = self.aggr(x, ptr=data.ptr) #computes per-graph embedding

            assert x.shape[0]==len(data.ptr)-1, f"pred.shape={pred.shape}"
        else:
            x = x.mean(dim=0).unsqueeze(0) #average across nodes

        return x
    
# Define the encoding model (RNN) architecture

class InputEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 bidirections=True):
        
        super().__init__()
        pass

