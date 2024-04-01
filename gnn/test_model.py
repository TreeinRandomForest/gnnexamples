from data import *
from model import *

g = read_graphs('data')
d = graph_to_data(g[0], 4)

'''
Average input nodes and project to output space
'''
model = GNNModel(in_node_dim=2,
                 out_dim=20,
                 layer_dims=[])

assert model(d).shape[1] == 20

model = GNNModel(in_node_dim=2,
                 out_dim=20,
                 layer_dims=[4, 8, 12])

assert model(d).shape[1] == 20