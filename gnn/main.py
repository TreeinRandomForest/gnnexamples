from model import *
from data import prepare_data_vocab, live_feat
from torch.nn.utils.rnn import unpack_sequence
from torch_geometric.data import Data

word_to_idx, idx_to_word, data, data_emb = prepare_data_vocab("data", 
                                                              func=live_feat, 
                                                              function_num=0)

in_dim = 8
hidden_dim = 16
num_layers = 7
bidirectional = True

int_dir = 2 if bidirectional else 1

bb_enc = InputEncoder(word_to_idx, 
                      in_dim=in_dim, 
                      hidden_dim=hidden_dim, 
                      num_layers=num_layers, 
                      bidirectional=bidirectional)

gnn = GNNModel(in_node_dim=int_dir*hidden_dim,
               out_dim=20,
               layer_dims=[16, 32])

#rnn forward pass
graph = data_emb[0]
edge_index, node_embs = graph
edge_index = torch.tensor(edge_index).T

out, h, c = bb_enc(node_embs)
node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])

assert node_feats.shape == (len(node_embs), int_dir*hidden_dim)

#gnn forward pass
graph = Data(edge_index=edge_index,
             x=node_feats)

assert graph.x.requires_grad

# gnn(graph) - fails because of faulty data#
# more nodes specified in edge_list than in node_feats 