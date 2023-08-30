import torch.nn as nn
from torch_geometric.nn import GCNConv

# Define the GNN architecture
class GNNModel(nn.Module):
    def __init__(self,
                 in_node_dim=2,
                 out_dim=20,
                 layer_dims=[16, 32]):

        super(GNNModel, self).__init__()

        layer_dims = [in_node_dim] + layer_dims

        self.conv_layers = nn.ModuleList({})

        for idx in range(1, len(layer_dims)):

            layer = self.GCNConv(in_channels=layer_dims[idx-1],
                                 out_channels=layer_dims[idx])
            
            self.conv_layers.append(layer)

        self.fc = nn.Linear(layer_dims[-1], out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.fc(x)
        x = torch.mean(x, dim=0, keepdim=True) #graph logits
        
        return x