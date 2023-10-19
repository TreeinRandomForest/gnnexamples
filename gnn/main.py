from model import *
from data import prepare_data_vocab, live_feat
from torch.nn.utils.rnn import unpack_sequence
from torch_geometric.data import Data

#read data
#word_to_idx: {'*': 0, '+': 1, '-': 2, '/': 3, 'M': 4, 'PHI': 5}
#   mapping from unique instruction to integer

#idx_to_word: {0: '*', 1: '+', 2: '-', 3: '/', 4: 'M', 5: 'PHI'}
#   inverse mapping of word_to_idx

#data
#   one entry per ssa.c file
#   data[i][0] - edge list (two node ids per edge)
#   data[i][1] - list of basic blocks with raw instructions in each BB
#   data[i][2] - same as data[i][1] but with args removed
#   BUG (introduced by SRA) - len(data[i][1]) != len(data[i][2])

#data_emb
#   same as data but with instructions replaced by uniq ids specified
#   in word_to_idx

word_to_idx, idx_to_word, data, data_emb = prepare_data_vocab("data", 
                                                              func=live_feat, 
                                                              function_num=0)
assert len(data)==len(data_emb)
for i in range(len(data)):
    assert len(data[i][2])==len(data_emb[i][1])

#define InputEncoder (RNN) and GNN
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
 
gnn = GNNModel(in_dim=int_dir*hidden_dim,
               out_dim=20,
               layer_dims=[16, 32])

#prediction/forward pass
#Step 1: rnn forward pass
graph = data_emb[0]
edge_index, node_embs = graph
edge_index = torch.tensor(edge_index).T

out, h, c = bb_enc(node_embs)
node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])

assert node_feats.shape == (len(node_embs), int_dir*hidden_dim)

#Step 2: gnn forward pass
graph = Data(edge_index=edge_index,
             x=node_feats)

assert graph.x.requires_grad

out = gnn(graph)
print(out.shape)
# more nodes specified in edge_list than in node_feats 