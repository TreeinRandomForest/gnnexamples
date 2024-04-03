import numpy as np
from torch.utils.data import Dataset
import torch

def gen_data(n_examples=1000,
             start_low=0, start_high=10,
             delta_low=1, delta_high=5,
             len_low=5, len_high=30):

    data = []
    for n in range(n_examples):
        start = np.random.randint(low=start_low, high=start_high)
        delta = np.random.randint(low=delta_low, high=delta_high)
        length = np.random.randint(low=len_low, high=len_high)

        d = start + np.arange(length) * delta
        data.append(d)

    return data
        
    
def generate_random_edge_index(num_nodes, max_length_factor=2):

    # Maximum number of edges based on the maximum length factor
    max_num_edges = num_nodes * max_length_factor

    # Generate random edge indices
    edge_index = torch.randint(0, num_nodes, (2, max_num_edges))
    
    # Ensure no self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    # Limit the number of edges to avoid exceeding max_length_factor
    num_edges = edge_index.shape[1]
    if num_edges > max_num_edges:
        edge_index = edge_index[:, :max_num_edges]
    
    return edge_index

class CustomData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
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