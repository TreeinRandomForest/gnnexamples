import torch
import torch.nn as nn
from data import gen_data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence,\
    pad_sequence, \
    pad_packed_sequence

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = gen_data(len_low=2, len_high=6)


class CustomData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#data = pack_padded_sequence(pad_sequence([torch.from_numpy(d) for d in data], batch_first=True), lengths=[len(d) for d in data], batch_first=True, enforce_sorted=False)

batch_size = 2

def func(seq):
    print(type(seq))
    print(seq)
    seq = [torch.from_numpy(s).float() for s in seq]
    lengths = [len(s) for s in seq]
    
    padded_seq = pad_sequence(seq, batch_first=True).unsqueeze(-1) #this changes if use embeddings i.e. no unsqueeze but map to embeddings
    packed_padded_seq = pack_padded_sequence(padded_seq,
                                             lengths=lengths,
                                             batch_first=True,
                                             enforce_sorted=False)
    return packed_padded_seq
    
ds = CustomData(data) #dataset
dl = DataLoader(ds,
                batch_size=batch_size,
                collate_fn=func) 

layer = nn.LSTM(input_size=7,
                hidden_size=3,
                batch_first=True)

emb = nn.Embedding(num_embeddings=256,
                   embedding_dim=7)

for idx in range(0, len(ds), batch_size):
    print(idx, idx+batch_size)
    seq = ds[idx:idx+batch_size]
    seq = [torch.from_numpy(s).float() for s in seq]
    lengths = [len(s) for s in seq]
    
    padded_seq = pad_sequence(seq, batch_first=True).long()
    padded_seq = emb(padded_seq)

    packed_padded_seq = pack_padded_sequence(padded_seq,
                                             lengths=lengths,
                                             batch_first=True,
                                             enforce_sorted=False)

    out, (hn, cn) = layer(packed_padded_seq)
    out = pad_packed_sequence(out)
    #out = out[0].to('cpu')
