N_max = 200 #vocab size i.e. distinct number of ints/instructions

emb_dim = enc_in_dim = dec_in_dim = 7
enc_bidirectional = dec_bidirectional = True
D = int(enc_bidirectional) + 1

enc_hidden_dim = 13
enc_num_layers = 3

dec_hidden_dim = 17
dec_num_layers = 3

start_val_idx = -1
