import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,\
                                pack_sequence,\
                                pad_packed_sequence

from data import *
from config import *

d = gen_data()

'''TODO:
- clean code, add config file
- training loop - loop over data, backward pass, sgd, test on holdout set
- decoder: ints vs embeddings
'''

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
