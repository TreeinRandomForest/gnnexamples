import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,\
                                pack_sequence,\
                                pad_packed_sequence

from data import *

d = gen_data()

N_max = 200
word_to_idx = dict(zip(np.arange(N_max), np.arange(N_max)))

class InputEncoder(nn.Module):
    def __init__(self,
                 word_to_idx,
                 in_dim,
                 hidden_dim,
                 num_layers,
                 bidirectional=True):
        
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
                
    
    def forward(self, x):
        #x should be one array
        #dim should be (1, len of sequence, embedding size) since lstm has batch_first=True
        embs = torch.stack([self.emb(elem) for elem in x]).unsqueeze(0)
        
        out, (h_n, c_n) = self.lstm(embs)

        return out, h_n, c_n

    
x = torch.from_numpy(d[0])
hidden_dim = 13
D = 2

enc = InputEncoder(word_to_idx, 7, hidden_dim, 3)
out, h_n, c_n = enc(x)

#choice - experiment here
state = c_n.reshape(1, -1)
#state = (h_n + c_n).reshape(1, -1) #element-wise addition

#projection
proj_h = nn.Linear(in_features=state.shape[1],
                   out_features=2*hidden_dim) #sorry for hard-coding here too
proj_c = nn.Linear(in_features=state.shape[1],
                   out_features=2*hidden_dim)

state_h = proj_h(h_n.reshape(1, -1)).reshape(2, -1)
state_c = proj_c(c_n.reshape(1, -1)).reshape(2, -1)

dec = nn.LSTM(input_size=1, #feeding integers
              hidden_size = 13, #sorry for hard-coding
              num_layers = 1,
              batch_first=True,
              bidirectional=True)

start_val = -1

output_layer = nn.Linear(in_features=D*hidden_dim,
                         out_features=N_max) #c_n -> output probabilities

outs = []
for i in range(x.shape[0]-1):
    if i==0:
        start_val=-1
    else:
        start_val = x[i-1].item()
        
    out, (h_n, c_n) = dec(torch.tensor(start_val).unsqueeze(0).unsqueeze(1).float(), (state_h, state_c))

    pred = output_layer(out)
    outs.append(torch.argmax(pred, dim=1))

    
    
