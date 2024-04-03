import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pad_sequence,\
                                pack_sequence,\
                                pad_packed_sequence,\
                                pack_padded_sequence
from torch.nn import LeakyReLU
from torch_geometric.nn.aggr import MeanAggregation
from data import *
from config import *

d = gen_data()

'''TODO:
- clean code, add config file
- training loop - loop over data, backward pass, sgd, test on holdout set
- decoder: ints vs embeddings
'''


class GNNModel(nn.Module):
    def __init__(self,
                 in_dim=2,
                 out_dim=20,
                 layer_dims=[16, 32],
                 name="gnn",
                 dropout_prob=0.5):

        super().__init__()

        layer_dims = [in_dim] + layer_dims

        self.conv_layers = nn.ModuleList({})

        for idx in range(1, len(layer_dims)):

            layer = GCNConv(in_channels=layer_dims[idx-1],
                            out_channels=layer_dims[idx])
            
            self.conv_layers.append(layer)

        self.fc = nn.Linear(layer_dims[-1], out_dim)
        self.leaky_relu = LeakyReLU(negative_slope=0.01)
        #self.aggr = MeanAggregation()

    def forward(self, x, edge_index):
        for l in self.conv_layers:
            x = l(x, edge_index)
            x = self.leaky_relu(x)

        x = self.fc(x)
        
        '''
        if 'ptr' in data:
            x = self.aggr(x, ptr=data.ptr) #computes per-graph embedding

            assert x.shape[0]==len(data.ptr)-1, f"pred.shape={pred.shape}"
        else:
            x = x.mean(dim=0).unsqueeze(0) #average across nodes
        '''
        return x



class InputEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional=True):
        
        super().__init__()
        
        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
                
    
    def forward(self, x):
        #x should be one array
        #dim should be (1, len of sequence, embedding size) since lstm has batch_first=True
        embs = torch.stack([self.emb(elem) for elem in x]).unsqueeze(0)
        
        out, (h_n, c_n) = self.lstm(embs)

        return out, h_n, c_n

class AE_rnn(nn.Module):
        def __init__(self,
                batch_size     = 2,
                bidir          = True,
                num_layers_enc = 3,
                hidden_dim_enc = 5,
                num_layers_dec = 7,
                hidden_dim_dec = 9,
                emb_dim        = 11,
                N_max          = 256):

                super().__init__()

                self.dir            = 2 if bidir else 1
                self.num_layers_dec = num_layers_dec
                self.batch_size     = batch_size

                self.enc = nn.LSTM(input_size=emb_dim,
                        bidirectional=True,
                        num_layers= num_layers_enc,
                        hidden_size=hidden_dim_enc,
                        batch_first=True)

                self.enc = nn.LSTM(input_size=emb_dim,
                                bidirectional=bidir,
                                num_layers= num_layers_enc,
                                hidden_size=hidden_dim_enc,
                                batch_first=True)

                self.dec = nn.LSTM(input_size=1,
                        bidirectional=bidir,
                        num_layers= num_layers_dec,
                        hidden_size=hidden_dim_dec,
                        batch_first=True)

                self.emb = nn.Embedding(num_embeddings=N_max,
                                embedding_dim=emb_dim)

                self.proj1 = nn.Linear(in_features=self.dir*num_layers_enc*hidden_dim_enc,
                                      out_features=self.dir*num_layers_dec*hidden_dim_dec) 
                        
                self.proj2 = nn.Linear(in_features=self.dir*num_layers_enc*hidden_dim_enc,
                                      out_features=self.dir*num_layers_dec*hidden_dim_dec)

                self.pred  = nn.Linear(in_features=self.dir*hidden_dim_dec,
                                      out_features=N_max)
                
        def forward(self, seq, lengths):
                padded_seq_emb = self.emb(seq)
                packed_padded_seq = pack_padded_sequence(padded_seq_emb,
                                                        lengths=lengths,#.tolist(),
                                                        batch_first=True,
                                                        enforce_sorted=False)

                out_enc, (hn, cn) = self.enc(packed_padded_seq)
                #print(hn.shape)
                #print(hn.shape)
                hn = self.proj1(hn.permute(1,0,2).flatten(1)).reshape(self.batch_size,self.num_layers_dec*self.dir,-1).permute(1,0,2).contiguous()
                cn = self.proj2(cn.permute(1,0,2).flatten(1)).reshape(self.batch_size,self.num_layers_dec*self.dir,-1).permute(1,0,2).contiguous()
                #print(hn.shape)
                out_dec, (hn_dec, cn_dec) = self.dec(seq.unsqueeze(2).float(), (hn, cn))
                out_pred = self.pred(out_dec)
                return out_pred
                


class AE_gnnrnn(nn.Module):
        def __init__(self,
                batch_size     = 2,
                bidir          = True,
                num_layers_enc = 3,
                hidden_dim_enc = 5,
                num_layers_dec = 7,
                hidden_dim_dec = 9,
                emb_dim        = 11,
                N_max          = 256):

                super().__init__()

                self.dir            = 2 if bidir else 1
                self.num_layers_dec = num_layers_dec
                self.batch_size     = batch_size

                self.enc = nn.LSTM(input_size=emb_dim,
                        bidirectional=True,
                        num_layers= num_layers_enc,
                        hidden_size=hidden_dim_enc,
                        batch_first=True)

                self.enc = nn.LSTM(input_size=emb_dim,
                                bidirectional=bidir,
                                num_layers= num_layers_enc,
                                hidden_size=hidden_dim_enc,
                                batch_first=True)

                self.dec = nn.LSTM(input_size=1,
                        bidirectional=bidir,
                        num_layers= num_layers_dec,
                        hidden_size=hidden_dim_dec,
                        batch_first=True)

                self.emb = nn.Embedding(num_embeddings=N_max,
                                embedding_dim=emb_dim)


                self.gnn1 = GNNModel(in_dim=self.dir*num_layers_enc*hidden_dim_enc,
                                    out_dim=self.dir*num_layers_enc*hidden_dim_enc)
                
                self.gnn2 = GNNModel(in_dim=self.dir*num_layers_enc*hidden_dim_enc,
                                    out_dim=self.dir*num_layers_enc*hidden_dim_enc)

                self.proj1 = nn.Linear(in_features=self.dir*num_layers_enc*hidden_dim_enc,
                                      out_features=self.dir*num_layers_dec*hidden_dim_dec) 
                        
                self.proj2 = nn.Linear(in_features=self.dir*num_layers_enc*hidden_dim_enc,
                                      out_features=self.dir*num_layers_dec*hidden_dim_dec)

                self.pred  = nn.Linear(in_features=self.dir*hidden_dim_dec,
                                      out_features=N_max)
                
        def forward(self, seq, lengths, edge_index):
                padded_seq_emb = self.emb(seq)
                packed_padded_seq = pack_padded_sequence(padded_seq_emb,
                                                        lengths=lengths,#.tolist(),
                                                        batch_first=True,
                                                        enforce_sorted=False)

                out_enc, (hn, cn) = self.enc(packed_padded_seq)
                
                hn = self.gnn1(hn.permute(1,0,2).flatten(1), edge_index)
                cn = self.gnn2(cn.permute(1,0,2).flatten(1), edge_index)

                hn = self.proj1(hn).reshape(self.batch_size,self.num_layers_dec*self.dir,-1).permute(1,0,2).contiguous()
                cn = self.proj2(cn).reshape(self.batch_size,self.num_layers_dec*self.dir,-1).permute(1,0,2).contiguous()
                out_dec, (hn_dec, cn_dec) = self.dec(seq.unsqueeze(2).float(), (hn, cn))
                out_pred = self.pred(out_dec)
                return out_pred
                
