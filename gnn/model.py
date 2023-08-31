import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for l in self.conv_layers:
            x = l(x, edge_index)
            x = x.relu()

        x = self.fc(x)
        #x = torch.mean(x, dim=0, keepdim=True) #graph logits
        
        #don't aggregate here since data is generally a batch
        #x needs to be split by data.ptr and aggregated to return
        #tensor of shape (batch_size, x.shape[1])
        #agg in training loop in train.py        

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

