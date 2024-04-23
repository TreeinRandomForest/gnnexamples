from model import AutoEncoder_gnnrnn, AutoEncoder_rnn
from data import prepare_data_vocab, live_feat, batch_gnn_for_gpu, split_pp_into_sublists, split_train_test, count_occurrences
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json

word_to_idx, idx_to_word, data, data_emb = prepare_data_vocab("data", func=live_feat, function_num=0)




if torch.cuda.is_available():
    device = torch.device("cuda")  # Or "cuda:0" for the first GPU
else:
    device = torch.device("cpu")
batch_size = 64 #config['batch_size']
dataset = split_pp_into_sublists(data_emb, batch_size)
(train, test) = split_train_test(dataset)
train_gpu = batch_gnn_for_gpu(train, device, len(word_to_idx))
test_gpu = batch_gnn_for_gpu(test, device, len(word_to_idx))
weight = count_occurrences(train, word_to_idx, device)

# Read configuration from JSON file
with open('config.json') as f:
    configs = json.load(f)

ratio_mix = 0.5

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

    if is_gnn:
        ae = AutoEncoder_gnnrnn(batch_size     = batch_size,
                bidir          = True,
                num_layers_enc = num_layers_enc,
                hidden_dim_enc = hidden_dim_enc,
                num_layers_dec = num_layers_dec,
                hidden_dim_dec = hidden_dim_dec,
                layer_dims_gnn = layer_dims_gnn,
                emb_dim        = emb_dim,
                N_max          = N_max)
    else:
        ae = AutoEncoder_rnn(batch_size     = batch_size,
            bidir          = True,
            num_layers_enc = num_layers_enc,
            hidden_dim_enc = hidden_dim_enc,
            num_layers_dec = num_layers_dec,
            hidden_dim_dec = hidden_dim_dec,
            emb_dim        = emb_dim,
            N_max          = N_max)

    

    #device = torch.device("cpu")
    ae.to(device)


    


    criterion = nn.CrossEntropyLoss(ignore_index=len(word_to_idx), weight=weight)
    #data_emb_device = [(edge_index.to(device), [bb.to(device) for bb in node_embs ]) for (edge_index, node_embs) in data_emb] 
    optimizer = optim.Adam(ae.parameters(), lr=0.01)

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
            out = ae(padded_seq, padded_seq_dec, lengths, edge_index, mode="mix", ratio=0.5, ratio_mix=ratio_mix)
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
    axs[0,0].plot(loss_train)
    axs[0,0].set_ylabel("loss")
    axs[0,0].set_xlabel("epochs")
    axs[0,0].set_title("train")
    axs[1,0].plot(accu_train)
    axs[1,0].set_ylabel("Accuracy")
    axs[1,0].set_xlabel("epochs")
    axs[1,0].set_title("train")
    axs[0,1].plot(loss_test)
    axs[0,1].set_ylabel("loss")
    axs[0,1].set_xlabel("epochs")
    axs[0,1].set_title("test")
    axs[1,1].plot(accu_test)
    axs[1,1].set_ylabel("Accuracy")
    axs[1,1].set_xlabel("epochs")
    axs[1,1].set_title("test")
    plt.tight_layout()
    plt.savefig('AE_result_'+config_name+'.png')