import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import MeanAggregation
from torch.nn.utils.rnn import pad_sequence,\
                                pack_sequence,\
                                pad_packed_sequence
from torch_geometric.data import Data
from torch.nn.utils.rnn import unpack_sequence
import copy
from torch.nn import LeakyReLU
import torch.nn.utils.rnn as rnn  


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# Define the GNN architecture
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
        self.aggr = MeanAggregation()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for l in self.conv_layers:
            x = l(x, edge_index)
            x = self.leaky_relu(x)

        x = self.fc(x)
    
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
    
# Define the encoding model (RNN) architecture
class InputEncoder(nn.Module):
    def __init__(self,
                 word_to_idx,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional=True,
                 dropout_prob=0.5):
        
        super().__init__()
        
        self.word_to_idx = word_to_idx
        vocab_size = len(word_to_idx)

        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.leaky_relu = LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        #x should be per-node bb ins indices
        embs = pack_sequence([self.leaky_relu(self.emb(bb)) for bb in x], enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(embs)
        
        #out = self.leaky_relu(out) # Apply Leaky ReLU
        return out, h_n, c_n


class AutoEncoder(nn.Module):
    def __init__(self,
                 word_to_idx,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 output_size,
                 bidirectional=True,
                 dropout_prob=0.5):
        
        super().__init__()
        
        self.word_to_idx = word_to_idx
        vocab_size = len(word_to_idx)

        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        
        int_dir = 2 if bidirectional else 1
        
        self.lstm_decoder = nn.LSTM(input_size=hidden_dim*int_dir,
                                    hidden_size=4,#output_size,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=bidirectional)

        self.leaky_relu = LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        #x should be per-node bb ins indices
        embs = pack_sequence([self.leaky_relu(self.emb(bb)) for bb in x], enforce_sorted=False)
        intput_embs = [self.emb(bb) for bb in x]
        

        out, (h_n, c_n) = self.lstm(embs)
        node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])
        original_array = node_feats.detach().numpy()
        deep_copied_array = copy.deepcopy(original_array)
        out_act = torch.from_numpy(deep_copied_array)

        decoded_outputs_temps = []
        decoded_outputs       = []
        for i in range(len(x)):
            current_sequence = out_act[i].repeat(len(x[i]), 1)
            for j in range(len(x[i])):
                
                output, (hidden_state, cell_state) = self.lstm_decoder(current_sequence)
                #print(output.size())
                output_f = torch.vstack([k[-1] for k in (output.T)])
                #print(output_f.size())
                decoded_outputs_temps.append(output_f)
            decoded_outputs.append(decoded_outputs_temps)

        return decoded_outputs, intput_embs


class AutoEncoderHidden(nn.Module):
    def __init__(self,
                 word_to_idx,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 output_size,
                 bidirectional=True,
                 dropout_prob=0.5):
        
        super().__init__()
        
        self.word_to_idx = word_to_idx
        vocab_size = len(word_to_idx)

        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        
        int_dir = 2 if bidirectional else 1
        
        self.lstm_decoder = nn.LSTM(input_size=in_dim,
                                    hidden_size=hidden_dim,#output_size,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=bidirectional)

        self.leaky_relu = LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        #x should be per-node bb ins indices
        embs = pack_sequence([self.leaky_relu(self.emb(bb)) for bb in x], enforce_sorted=False)
        intput_embs = [self.emb(bb) for bb in x]
        

        out, (h_n, c_n) = self.lstm(embs)
        node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])
        original_array = node_feats.detach().numpy()
        deep_copied_array = copy.deepcopy(original_array)
        out_act = torch.from_numpy(deep_copied_array)

        decoded_outputs_temps = []
        decoded_outputs       = []
        for i in range(len(x)):
            current_sequence = out_act[i].repeat(len(x[i]), 1)
            for j in range(len(x[i])):
                
                output, (hidden_state, cell_state) = self.lstm_decoder(current_sequence)
                #print(output.size())
                output_f = torch.vstack([k[-1] for k in (output.T)])
                #print(output_f.size())
                decoded_outputs_temps.append(output_f)
            decoded_outputs.append(decoded_outputs_temps)

        return decoded_outputs, intput_embs


class IntegratedModel(nn.Module):

    def __init__(self, 
                 word_to_idx,
                 in_dim_lstm,
                 hidden_dim,
                 num_layers,
                 bidirectional=True,
                 out_dim=20,
                 layer_dims=[16, 32]):
        
        super().__init__()
        self.lstm_encoder = InputEncoder(word_to_idx, 
                            in_dim=in_dim_lstm, 
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional)
        
        self.gnn_model = GNNModel(in_dim=(int(bidirectional)+1)*hidden_dim,
                         out_dim=out_dim,
                         layer_dims=[16, 32])
    
    def forward(self, node_sequences, edge_index):
        # node_sequences: shape (num_nodes, seq_length, lstm_input_size)
        # edge_index: shape (2, num_edges)
        out, h, c = self.lstm_encoder(node_sequences)
        node_feats = torch.vstack([k[-1] for k in unpack_sequence(out)])
        original_array = node_feats.detach().numpy()
        deep_copied_array = copy.deepcopy(original_array)
        out_act = torch.from_numpy(deep_copied_array)
        node_feats[:,16:] = -1
        graph = Data(edge_index=edge_index,
                        x=node_feats)
        gnn_output = self.gnn_model(graph)
        
        return gnn_output, out_act
    

class AE_rnn(nn.Module):
    def __init__(self,
                 device,
                 word_to_idx,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional=True):
        D = 2
        N_max = len(word_to_idx)
        super().__init__()

        self.device = device
        self.word_to_idx = word_to_idx #identity mapping
        vocab_size = len(word_to_idx)

        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        self.emb2 = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
                
        self.proj1 = nn.Linear(in_features=num_layers*2*hidden_dim,#state.shape[1],
                   out_features=2*hidden_dim) #sorry for hard-coding here too
        
        self.proj2 = nn.Linear(in_features=num_layers*2*hidden_dim,#state.shape[1],
                   out_features=2*hidden_dim)

        self.dec   = nn.LSTM(input_size=1, #feeding integers
              hidden_size = 13, #sorry for hard-coding
              num_layers = 1,
              batch_first=True,
              bidirectional=True)
        self.out   = nn.Linear(in_features=D*hidden_dim,
                         out_features=N_max) #c_n -> output probabilities
    def forward(self, x):
        #x should be one array
        #dim should be (1, len of sequence, embedding size) since lstm has batch_first=True
        
        embs = torch.stack([self.emb(elem) for elem in x]).unsqueeze(0)
        

        out, (h_n, c_n) = self.lstm(embs)


        

        state = c_n.reshape(1, -1)
        
        state_h = self.proj1(h_n.reshape(1, -1)).reshape(2, -1)
        state_c = self.proj2(c_n.reshape(1, -1)).reshape(2, -1)

        

        start_val = -1
        outs = []
        for i in range(x.shape[0]):
            if i==0:
                start_val=-1
            else:
                start_val = x[i-1].item()

            if i==0:
                out, (h_n, c_n) = self.dec(torch.tensor(start_val).unsqueeze(0).unsqueeze(1).float().to(self.device), (state_h, state_c))
            else:
                out, (h_n, c_n) = self.dec(torch.tensor(start_val).unsqueeze(0).unsqueeze(1).float().to(self.device))

            pred = self.out(out)
            #outs.append(torch.argmax(pred, dim=1))
            outs.append(pred.squeeze())
        #print(outs)
        #return torch.stack([self.emb(elem) for elem in outs]).squeeze(), embs.squeeze()
        return torch.stack(outs), embs.squeeze()
    


class AE_gnnrnn(nn.Module):
    def __init__(self,
                 word_to_idx,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional=True):
        D = 2
        N_max = len(word_to_idx)
        super().__init__()
        
        self.word_to_idx = word_to_idx #identity mapping
        vocab_size = len(word_to_idx)

        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        
        self.gnn_model_h = GNNModel(in_dim=2*hidden_dim,
                         out_dim=2*hidden_dim,
                         layer_dims=[16, 32])
        
        self.gnn_model_c = GNNModel(in_dim=2*hidden_dim,
                         out_dim=2*hidden_dim,
                         layer_dims=[16, 32])

        self.proj1 = nn.Linear(in_features=num_layers*2*hidden_dim,#state.shape[1],
                   out_features=2*hidden_dim) #sorry for hard-coding here too
        
        self.proj2 = nn.Linear(in_features=num_layers*2*hidden_dim,#state.shape[1],
                   out_features=2*hidden_dim)

        self.dec   = nn.LSTM(input_size=1, #feeding integers
              hidden_size = 13, #sorry for hard-coding
              num_layers = 1,
              batch_first=True,
              bidirectional=True)
        self.out   = nn.Linear(in_features=D*hidden_dim,
                         out_features=N_max) #c_n -> output probabilities
    def forward(self, x, edge_index):
        #x should be one array
        #dim should be (1, len of sequence, embedding size) since lstm has batch_first=True
        state_h_list = []
        state_c_list = []
        state_list   = []
        for bb in x:
            embs = torch.stack([self.emb(elem) for elem in bb]).unsqueeze(0)
            out, (h_n, c_n) = self.lstm(embs)
            state_h_bb = self.proj1(h_n.reshape(1, -1))
            state_c_bb = self.proj2(c_n.reshape(1, -1))
            state_h_list.append(state_h_bb)
            state_c_list.append(state_c_bb)
#            state_list.append(torch.stack([state_h_bb, state_c_bb]).reshape(1,-1))

        state_h_ten = torch.vstack(state_h_list)
        state_c_ten = torch.vstack(state_c_list)
        #state_ten   = torch.vstack(state_list)

        
        graph_h = Data(edge_index=edge_index,
                        x=state_h_ten)
        
        graph_c = Data(edge_index=edge_index,
                        x=state_c_ten)
        
        #graph = Data(edge_index=edge_index,
#                        x=state_ten)
        
        state_h_graph = self.gnn_model_h(graph_h)

        state_c_graph = self.gnn_model_c(graph_c)

        #state_graph   = self.gnn_model(graph)

        outs = []
        for k in range(state_h_graph.shape[0]):
            outs_bb = []
            start_val = -1
            #print(state_graph.shape)
            #print(state_h_graph.shape)

            #state_h_graph = state_graph[:, 0:int(state_graph.shape[1]/2)]
            #state_c_graph = state_graph[:, int(state_graph.shape[1]/2):int(state_graph.shape[1])]
            #print(state_h_graph.shape)
            state_h = state_h_graph[k].reshape(2, -1)
            state_c = state_c_graph[k].reshape(2, -1)
           
            for i in range(x[k].shape[0]):
                if i==0:
                    start_val=-1
                else:
                    start_val = x[k][i-1].item()

                if i==0:
                    out, (h_n, c_n) = self.dec(torch.tensor(start_val).unsqueeze(0).unsqueeze(1).float(), (state_h, state_c))
                else:
                    out, (h_n, c_n) = self.dec(torch.tensor(start_val).unsqueeze(0).unsqueeze(1).float())

                pred = self.out(out)
                #outs.append(torch.argmax(pred, dim=1))
                outs_bb.append(pred.squeeze())
            #print("out_bb", torch.stack(outs_bb).shape)
            outs.append(torch.stack(outs_bb))
        #print(outs)
        #return torch.stack([self.emb(elem) for elem in outs]).squeeze(), embs.squeeze()
        return outs
    


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.data import Data

class AE_gnnrnn_gpu(nn.Module):
    def __init__(self,
                 word_to_idx,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional=True):
        D = 2
        N_max = len(word_to_idx)
        super().__init__()
        
        self.word_to_idx = word_to_idx # identity mapping
        vocab_size = len(word_to_idx)

        self.emb = nn.Embedding(num_embeddings=vocab_size+1,
                                embedding_dim=in_dim,
                                padding_idx=-1)
        
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        
        self.gnn_model_h = GNNModel(in_dim=2*hidden_dim,
                         out_dim=2*hidden_dim,
                         layer_dims=[16, 32])
        
        self.gnn_model_c = GNNModel(in_dim=2*hidden_dim,
                         out_dim=2*hidden_dim,
                         layer_dims=[16, 32])

        self.proj1 = nn.Linear(in_features=num_layers*2*hidden_dim,
                               out_features=2*hidden_dim)
        
        self.proj2 = nn.Linear(in_features=num_layers*2*hidden_dim,
                               out_features=2*hidden_dim)

        self.dec = nn.LSTM(input_size=1,  # feeding integers
                           hidden_size=13,  # sorry for hard-coding
                           num_layers=1,
                           batch_first=True,
                           bidirectional=True)
        
        self.out = nn.Linear(in_features=D*hidden_dim,
                             out_features=N_max)  # c_n -> output probabilities

    def forward(self, x, edge_index):
        # x should be one array
        # dim should be (1, len of sequence, embedding size) since lstm has batch_first=True
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True)
        lengths = [len(seq) for seq in x]
        packed_input = pack_padded_sequence(x_padded, lengths, batch_first=True, enforce_sorted=False)
        packed_embs = self.emb(packed_input.data)
        packed_out, (h_n, c_n) = self.lstm(packed_embs)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        last_hidden_states = torch.stack([out[i, length - 1, :] for i, length in enumerate(lengths)])
        state_h_list = self.proj1(last_hidden_states)
        state_c_list = self.proj2(last_hidden_states)
        state_h_ten = torch.vstack(state_h_list)
        state_c_ten = torch.vstack(state_c_list)

        graph_h = Data(edge_index=edge_index, x=state_h_ten)
        graph_c = Data(edge_index=edge_index, x=state_c_ten)
        
        state_h_graph = self.gnn_model_h(graph_h)
        state_c_graph = self.gnn_model_c(graph_c)

        outs_bb = []
        start_val = -1
        for k in range(state_h_graph.shape[0]):
            state_h = state_h_graph[k].reshape(2, -1)
            state_c = state_c_graph[k].reshape(2, -1)

            if k > 0:
                start_val = x[k - 1][-1].item()

            inputs = torch.tensor([[start_val]]).unsqueeze(2).float().to(x.device)
            if k == 0:
                inputs = inputs.expand(-1, x[k].shape[0], -1)

            out, (h_n, c_n) = self.dec(inputs, (state_h, state_c))
            pred = self.out(out.squeeze())
            outs_bb.append(pred)

        return torch.stack(outs_bb)



class AutoEncoder_rnn(nn.Module):
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
                


class AutoEncoder_gnnrnn(nn.Module):
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

                hn = self.proj1(hn).reshape(len(lengths),self.num_layers_dec*self.dir,-1).permute(1,0,2).contiguous()
                cn = self.proj2(cn).reshape(len(lengths),self.num_layers_dec*self.dir,-1).permute(1,0,2).contiguous()
                out_dec, (hn_dec, cn_dec) = self.dec(seq.unsqueeze(2).float(), (hn, cn))
                out_pred = self.pred(out_dec)
                return out_pred
                
