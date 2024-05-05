from model import AutoEncoder_gnnrnn, AutoEncoder_rnn, AutoEncoder_gnngatrnn
from data import prepare_data_vocab, live_feat, batch_gnn_for_gpu, split_pp_into_sublists, split_train_test, count_occurrences
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from utils import calculate_variable
import pickle

if len(sys.argv) < 2:
    filename = 'config.json'
else:
    filename = sys.argv[1]

with open('data.pickle', 'rb') as f:
    [word_to_idx, idx_to_word, data, data_emb] = pickle.load(f)


new_data_emb = []
for g in data_emb:
    num_nodes = len(g[1])
    if g[0].max() < num_nodes:
        new_data_emb.append(g)
        
data_emb = new_data_emb

print(filename)

if torch.cuda.is_available():
    device = torch.device("cuda")  # Or "cuda:0" for the first GPU
    drop_rate = calculate_variable()
else:
    device = torch.device("cpu")
    drop_rate = 0

print("DROP RATE: ", drop_rate)

batch_size = 256 #config['batch_size']
dataset = split_pp_into_sublists(data_emb, batch_size)
(train, test) = split_train_test(dataset, drop_rate=drop_rate)
train_gpu = batch_gnn_for_gpu(train, device, len(word_to_idx))
test_gpu = batch_gnn_for_gpu(test, device, len(word_to_idx))
weight = count_occurrences(train, word_to_idx, device)

# Read configuration from JSON file
with open(filename) as f:
    configs = json.load(f)

ratio_mix = 0.5
ratio = 0.5
mode = 'mix'
is_gat = False
num_layers_gnn = 1

for config_name, config in configs.items():
    # Load configuration variables to exact names
    is_gnn = config["is_gnn"]
    bidir = config['bidir']
    
    num_layers_enc = config['num_layers_enc']
    hidden_dim_enc = config['hidden_dim_enc']
    num_layers_dec = config['num_layers_dec']
    hidden_dim_dec = config['hidden_dim_dec']
    emb_dim = config['emb_dim']
    layer_dims_gnn = config['layer_dims_gnn']
    epoch_num = config["epoch_num"]

    N_max    = len(word_to_idx)+1
    dir = 2 if bidir==True else 1

    if 'ratio_mix' in config:
        ratio_mix = config['ratio_mix']

    if 'ratio' in config:
        ratio = config['ratio']

    if 'mode' in config:
        mode = config['mode']

    if 'is_gat' in config:
        is_gat = config['is_gat']
        
    if 'num_layers_gnn' in config:
        num_layers_gnn = config['num_layers_gnn']
    
    if is_gnn:
        if is_gat:
            ae = AutoEncoder_gnngatrnn(batch_size     = batch_size,
            bidir          = True,
            num_layers_enc = num_layers_enc,
            hidden_dim_enc = hidden_dim_enc,
            num_layers_dec = num_layers_dec,
            hidden_dim_dec = hidden_dim_dec,
            num_layers_gnn = num_layers_gnn,
            emb_dim        = emb_dim,
            N_max          = N_max)
            print("GAT")
        else:
            ae = AutoEncoder_gnnrnn(batch_size     = batch_size,
                    bidir          = bidir,
                    num_layers_enc = num_layers_enc,
                    hidden_dim_enc = hidden_dim_enc,
                    num_layers_dec = num_layers_dec,
                    hidden_dim_dec = hidden_dim_dec,
                    layer_dims_gnn = layer_dims_gnn,
                    emb_dim        = emb_dim,
                    N_max          = N_max)
            print("GNN")
        ae.to(device)
        #optimizer = optim.Adam(ae.parameters(), lr=0.01, weight_decay=0.001)
    else:
        ae = AutoEncoder_rnn(batch_size     = batch_size,
            bidir          = bidir,
            num_layers_enc = num_layers_enc,
            hidden_dim_enc = hidden_dim_enc,
            num_layers_dec = num_layers_dec,
            hidden_dim_dec = hidden_dim_dec,
            emb_dim        = emb_dim,
            N_max          = N_max)
        print("LSTM")
        ae.to(device)
        #optimizer = optim.Adam(ae.parameters(), lr=0.01)

    #device = torch.device("cpu")
    


    

    optimizer = optim.Adam(ae.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=len(word_to_idx), weight=weight)
    #data_emb_device = [(edge_index.to(device), [bb.to(device) for bb in node_embs ]) for (edge_index, node_embs) in data_emb] 
    

    #hooks = register_hooks(ae)

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 

    #optimizer = optim.SGD(ae.parameters(), lr=0.1, momentum=0.9)
    #scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)


    loss_train = []
    accu_train = []
    loss_test = []
    accu_test = []
    for epoch in range(epoch_num):
        total_loss = 0
        j = 0
        cntBatch = 0
        for (padded_seq, padded_seq_dec, lengths, edge_index) in train_gpu:
            cntBatch += 1
            #print("seq ->", padded_seq.shape)
            #print("lengths -> ", min(lengths), ":", max(lengths))
            out = ae(padded_seq, padded_seq_dec, lengths, edge_index, mode=mode, ratio=ratio, ratio_mix=ratio_mix)
            #print(out[:,-1,:])
            loss = criterion(out.flatten(0).reshape(-1,N_max), padded_seq.flatten())
            #print(out.flatten(0).reshape(-1,N_max).argmax(dim=1).to("cpu").tolist())
            #print(padded_seq.flatten().to("cpu").tolist())
            #print("-------------------------------------------------")
            total_loss  += loss.item()
            j += np.sum(lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
        #for hook in hooks:
        #   hook.remove()  # Remove previous hooks
        #hooks = register_hooks(ae)
        # Optionally, you can print the learning rate to monitor its changes
            #print("Learning Rate:", optimizer.param_groups[0]['lr'])
        if epoch%5==0 or (epoch+1)==epoch_num:
            print(f'For {j} Tokens List in Train set: Epoch {epoch+1}/{epoch_num}, Average Loss: {total_loss/cntBatch:.5f}, Acuuracy: {np.exp(-total_loss/cntBatch)*100:.5f}', end='\n')
            cntBatch_test = 0
            total_loss_test = 0
            j_test = 0
            for (padded_seq, padded_seq_dec, lengths, edge_index) in test_gpu:
                j_test += np.sum(lengths)
                cntBatch_test += 1
                out = ae(padded_seq, padded_seq_dec, lengths, edge_index, mode="recursive", ratio=1)
                loss = criterion(out.flatten(0).reshape(-1,N_max), padded_seq.flatten())
                total_loss_test  += loss.item()
            loss_test.append(total_loss_test/cntBatch_test)
            accu_test.append(np.exp(-total_loss_test/cntBatch_test)*100)
            print(f'For {j_test} Tokens List in Test set: Epoch {epoch+1}/{epoch_num}, Average Loss: {total_loss_test/cntBatch_test:.5f}, Acuuracy: {np.exp(-total_loss_test/cntBatch_test)*100:.5f}', end='\n')
            print("-----------------------------------------------------------------------------------")
        else:
            print(f'For {j} Tokens List in Train set: Epoch {epoch+1}/{epoch_num}, Average Loss: {total_loss/cntBatch:.5f}, Acuuracy: {np.exp(-total_loss/cntBatch)*100:.5f}', end='\r')
        loss_train.append(total_loss/cntBatch)
        accu_train.append(np.exp(-total_loss/cntBatch)*100)

    fig, axs = plt.subplots(2, 2, figsize=(10,6)) 
    x = [i*5 for i in range(len(accu_test))]
    axs[0,0].plot(loss_train)
    axs[0,0].set_ylabel("loss")
    axs[0,0].set_xlabel("epochs")
    axs[0,0].set_title("train")
    axs[1,0].plot(accu_train)
    axs[1,0].set_ylabel("Accuracy")
    axs[1,0].set_xlabel("epochs")
    axs[1,0].set_title("train")
    axs[0,1].plot(x, loss_test)
    axs[0,1].set_ylabel("loss")
    axs[0,1].set_xlabel("epochs")
    axs[0,1].set_title("test")
    axs[1,1].plot(x, accu_test)
    axs[1,1].set_ylabel("Accuracy")
    axs[1,1].set_xlabel("epochs")
    axs[1,1].set_title("test")
    plt.tight_layout()
    plt.savefig('AE_result_'+config_name+'.png')
