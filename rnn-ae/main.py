import torch.nn as nn

from data import gen_data
from model import *
from config import *

'''
Choices:

Encoder:
integers - raw || encodings (N_max)

Decoder:
integers - raw || encodings (N_max)

Information passing:
There are L (layers) x 2 (D, bidirectional) x 2 (hidden and cell) vectors for last T
Output shape for h_n/c_n: (D*num_layers, N_batch, H_out_enc)

LSTM forward takes optional inputs:
h_0/c_0 of shape (D*num_layers, N_batch, H_out_dec)

Need projection layer from H_out_enc -> H_out_dec
'''

if __name__=='__main__':
    #initialize
    d = gen_data()
    x = d[5]

    #share embeddings between encoder and decoder
    emb = nn.Embedding(num_embeddings=N_max+1,
                       embedding_dim=emb_dim
                       )
    
    enc = nn.LSTM(input_size = enc_in_dim,
                  hidden_size = enc_hidden_dim,
                  num_layers = enc_num_layers,
                  batch_first=True,
                  bidirectional=enc_bidirectional)

    dec = nn.LSTM(input_size = dec_in_dim,
                  hidden_size = dec_hidden_dim,
                  num_layers = dec_num_layers,
                  batch_first=True,
                  bidirectional=dec_bidirectional)

    proj_h = nn.Linear(in_features=D * enc_num_layers * enc_hidden_dim,
                       out_features=D * dec_num_layers * dec_hidden_dim)

    proj_c = nn.Linear(in_features=D * enc_num_layers * enc_hidden_dim,
                       out_features=D * dec_num_layers * dec_hidden_dim)
    
    #forward pass through encoder
    embs = emb(torch.from_numpy(x))
    out, (h_n, c_n) = enc(embs)

    #initialize hidden/cell states for decoder
    dec_h_0 = proj_h(h_n.flatten()).reshape(D*dec_num_layers, dec_hidden_dim)
    dec_c_0 = proj_h(c_n.flatten()).reshape(D*dec_num_layers, dec_hidden_dim)

    
    
    
